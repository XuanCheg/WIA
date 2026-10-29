"""Run with pytest: dual-arm dispatch, metric contracts and public readback isolation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.action_compiler import MacroLimits, PrimitiveResolution, compile_action, parameterized_schema
from core.action_interface import BackendCapabilities
from core.action_units import RIGHT_ARM_ATOMS
from core.observations import AgentRobotState, AgentActionFeedback
from wia.runtime.sim.esi_arm import ArmControlConfig
from wia.runtime.sim.esi_dual_arm import R1DualManipulationController
from harness_integration.test_esi_arm import InverseKinematicsController
from harness_integration.test_esi_base import JointController, HolonomicBaseJointController


def dual_fixture():
    groups = {'base': 3, 'trunk': 4, 'arm_left': 6, 'gripper_left': 2,
              'arm_right': 6, 'gripper_right': 2}
    dofs = {}
    start = 0
    for name, count in groups.items():
        dofs[name] = np.arange(start, start + count)
        start += count
    joints_by_name = {side + str(i): SimpleNamespace(joint_type='JOINT_PRISMATIC')
                      for side in ('left', 'right') for i in range(2)}

    class NamedArray(np.ndarray):
        def __getitem__(self, key):
            return joints_by_name[key] if isinstance(key, str) else super().__getitem__(key)

    class Robot:
        name = 'dual_test'
        action_dim = start
        grasping_mode = 'assisted'
        def __init__(self):
            self.joints = np.zeros(start).view(NamedArray)
            for side, y in [('left', .2), ('right', -.2)]:
                self.joints[dofs['arm_' + side]] = [.45, y, 1.2, 0, 0, 0]
                self.joints[dofs['gripper_' + side]] = .03
            self.controllers = {n: (HolonomicBaseJointController(d) if n == 'base' else JointController(d)) for n, d in dofs.items()}
            permutation = np.random.default_rng(3).permutation(start)
            self.controller_action_idx = {n: permutation[d] for n, d in dofs.items()}
            self.arm_control_idx = {s: dofs['arm_' + s] for s in ('left', 'right')}
            self.finger_joint_names = {s: [s+'0', s+'1'] for s in ('left', 'right')}
            self._ag_obj_in_hand = dict(left=None, right=None)
            self._ag_obj_constraints = dict(left=None, right=None)
            self._ag_release_counter = dict(left=None, right=None)
            self._controller_config = {n: {'name': type(c).__name__} for n,c in self.controllers.items()}
            self._default_arm_ik_controller_configs = {s: {'name':'InverseKinematicsController'} for s in ('left','right')}
            lo, hi = np.full(start, -10.), np.full(start, 10.)
            for s in ('left','right'): lo[dofs['gripper_'+s]], hi[dofs['gripper_'+s]] = 0, .05
            self.control_limits = {'position':(lo,hi), 'velocity':(np.full(start,-5.),np.full(start,5.))}
        def reload_controllers(self, configs):
            for side in ('left', 'right'):
                c = InverseKinematicsController(dofs['arm_'+side])
                c.task_name = 'eef_'+side
                c.compute_no_op_action = lambda control, s=side: self.joints[dofs['arm_'+s]].copy()
                self.controllers['arm_'+side] = c
            self._controller_config = configs
        def get_joint_positions(self): return self.joints.copy()
        def get_position_orientation(self): return np.zeros(3), np.array([0.,0.,0.,1.])
        def get_eef_pose(self, side):
            q = self.joints[dofs['arm_'+side]]
            return q[:3].copy(), Rotation.from_rotvec(q[3:]).as_quat()
        def get_control_dict(self):
            d = {'joint_position':self.joints}
            for s in ('left','right'):
                p,q=self.get_eef_pose(s)
                d.update({f'eef_{s}_pos_relative':p, f'eef_{s}_quat_relative':q})
            return d
        def is_grasping(self, side): return -1

    robot = Robot()
    class Env:
        env_config={'action_frequency':30}
        fail=False
        def step(self, actions):
            if self.fail: raise RuntimeError('test interruption')
            for n, idx in dofs.items():
                c=robot.controllers[n]
                guard=getattr(c.compute_control, '__self__', None) if hasattr(c,'compute_control') else None
                hold=getattr(guard, 'explicit_joint_hold', None)
                robot.joints[idx] = hold if hold is not None else actions[robot.name][robot.controller_action_idx[n]]
    env=Env()
    c=R1DualManipulationController(SimpleNamespace(robot=robot,env=env),
        arm_config=replace(ArmControlConfig.load(),initial_trunk_joint_positions_rad=None))
    return c,robot,env


def test_right_metric_motion_holds_left_and_left_motion_holds_right():
    c,r,_=dual_fixture()
    left=r.get_eef_pose('left')[0].copy()
    right=r.get_eef_pose('right')[0].copy()
    result=c.execute_metric('RIGHT_MV_UP', .02)
    assert result['action']=='RIGHT_MV_UP' and result['completed']
    assert result['working_arm']=='right'
    assert r.get_eef_pose('right')[0][2]-right[2] == pytest.approx(.02,abs=.003)
    np.testing.assert_allclose(r.get_eef_pose('left')[0],left)
    held=r.get_eef_pose('right')[0].copy()
    assert c.execute('MV_UP')['completed']
    np.testing.assert_allclose(r.get_eef_pose('right')[0],held)
    assert c.working_arm=='left'


def test_right_gripper_and_public_readback_are_independent():
    c,r,_=dual_fixture()
    assert c.execute('RIGHT_RELEASE')['completed']
    assert c.execute('RIGHT_GRASP')['completed']
    state=c.controller_state()
    public=AgentRobotState.from_controller_state(state)
    assert public.right_arm.gripper_width_m==pytest.approx(0)
    assert public.gripper_width_m==pytest.approx(.06)
    assert public.right_arm.gripper_command_name=='GRASP'
    assert public.gripper_command_name=='HOLD'
    assert public.right_arm.holding_estimate is None
    assert AgentRobotState.from_payload(public.to_dict())==public
    feedback=AgentActionFeedback.from_result(c.execute('RIGHT_RELEASE'),state=c.controller_state())
    assert feedback.action=='RIGHT_RELEASE'
    assert feedback.estimated_motion.get('gripper_width_m')==pytest.approx(.1)
    assert 'object_name' not in str(public.to_dict())


def test_right_exception_restores_arm_context_and_inactive_guard():
    c,r,env=dual_fixture();env.fail=True
    with pytest.raises(RuntimeError,match='test interruption'):
        c.execute('RIGHT_MV_UP')
    assert c.working_arm=='left'
    assert c.ik_guard.explicit_joint_hold is None
    assert c._arm_contexts['right']['ik_guard'].explicit_joint_hold is not None


@pytest.mark.parametrize('token',RIGHT_ARM_ATOMS)
def test_right_schema_bounds_and_capability_gate(token):
    from core.action_compiler import magnitude_field
    import jsonschema
    capabilities=BackendCapabilities(right_ee_translation=True,right_ee_rotation=True,right_ee_tilt=True,right_gripper=True)
    resolution=PrimitiveResolution(arm_translation_m=.005,arm_rotation_deg=5)
    limits=MacroLimits(max_arm_primitives=16)
    field=magnitude_field(token)
    payload={'action':token, **({field:.02 if field=='distance_m' else 5} if field else {})}
    plan=compile_action(payload,capabilities,resolution,limits)
    assert plan.primitive_token==token
    jsonschema.validate(payload,parameterized_schema(capabilities.available_actions,limits,resolution))
    with pytest.raises(ValueError):compile_action(payload,BackendCapabilities(ee_translation=True,gripper=True),resolution,limits)
