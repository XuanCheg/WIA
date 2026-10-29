"""Verify native R1 trunk lower endpoints; run with run_esi_robot CLI arguments.

Example: python -m wia.runtime.commands.verify_r1_trunk_floor --web ...
This isolated diagnostic replaces the web loop, never an existing live scene.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def main():
    from wia.runtime.commands.run_esi_robot import main as robot_main
    import wia.runtime.sim.esi_web as web
    from core.action_compiler import ParameterizedAction
    args = sys.argv[1:]
    vertical_only = '--vertical-only' in args
    if vertical_only: args.remove('--vertical-only')
    output = Path(args[args.index('--output-dir') + 1])

    def verify(session, metadata, controller, executor, **kwargs):
        trunk = controller.trunk_controller
        model = trunk.kinematics
        initial = trunk.joints().copy()
        position, orientation = model.pose(initial)
        records = []
        report = dict(schema='r1_trunk_floor_v1', native_lower=trunk.lower.tolist(),
                      native_upper=trunk.upper.tolist(), initial_joints=initial.tolist(),
                      original_config=vars(trunk.config), records=records, passed=False)
        def save():
            (output/'trunk_floor_verification.json').write_text(json.dumps(report, indent=2)+'\n')
        def measured():
            q = trunk.joints()
            bp, bq = controller.robot.get_position_orientation()
            tp, _ = controller.robot.links['torso_link4'].get_position_orientation()
            def arr(x): return x.detach().cpu().numpy()
            native = Rotation.from_quat(arr(bq)).inv().apply(arr(tp)-arr(bp))
            return dict(joints=q.tolist(), nominal=model.pose(q)[0].tolist(),
                        installed_fk_m=trunk.measured_height(q), native_base_position=native.tolist())
        def command(target):
            before = measured()
            delta = target - before['nominal'][2]
            result = executor.execute_macro(ParameterizedAction(action='TRUNK_UP' if delta > 0 else 'TRUNK_DOWN', distance_m=round(abs(delta), 12)),
                resolution=executor.primitive_resolution, limits=executor.macro_limits)
            row = dict(target_control_height_m=target, before=before, after=measured(), result=result)
            records.append(row); save()
            print('[trunk-floor] '+json.dumps(dict(target=target, after=row['after'], reason=result.get('reason'))), flush=True)
            if not result.get('completed'): raise RuntimeError('Native trunk command failed: '+str(result.get('reason')))
        try:
            results = {}
            for margin, lower, upper in [('raw',trunk.lower-trunk.config.joint_limit_margin_rad,trunk.upper+trunk.config.joint_limit_margin_rad),('guarded',trunk.lower,trunk.upper)]:
                for fixed in [True,False]:
                    def constraint(q):
                        p,r=model.pose(q)
                        return np.r_[([p[0]-position[0]] if fixed else []), (orientation.inv()*r).as_rotvec()[1], q[3]-initial[3]]
                    result=minimize(lambda q:model.pose(q)[0][2],initial,bounds=list(zip(lower,upper)),
                        constraints={'type':'eq','fun':constraint},method='SLSQP',options={'ftol':1e-12,'maxiter':300})
                    if not result.success: raise RuntimeError(result.message)
                    results[margin+('_fixed_x' if fixed else '_free_x')]=dict(height=float(result.fun), joints=result.x.tolist(), position=model.pose(result.x)[0].tolist())
            report['kinematic_endpoints']=results;save()
            if not vertical_only: trunk.config=replace(trunk.config,minimum_height_m=.49)
            # Validate ordinary vertical commands to a rounded-up guarded endpoint.
            fixed_target=float(np.ceil((results['guarded_fixed_x']['height']+.001)*1000)/1000)
            while model.pose(trunk.joints())[0][2]-fixed_target > .30:
                command(model.pose(trunk.joints())[0][2]-.30)
            command(fixed_target)
            if vertical_only:
                command(model.pose(trunk.joints())[0][2]+.03)
                command(model.pose(trunk.joints())[0][2]-.03)
                report['passed']=True
                report['validated_scope']='fixed-horizontal upright trunk motion; free-X endpoint excluded after native self-collision'
                return
            # Separately validate the free-horizontal endpoint with native joint motion.
            # This is diagnostic only: ordinary TRUNK actions still preserve horizontal position.
            free_q=np.array(results['guarded_free_x']['joints'])
            free_q[1]+=.003
            free_q[2]+=.003
            free_q[0]-=.002
            free_q[2]-=.002  # Preserve upright pitch while staying inside the upper joint bound.
            free_position=model.pose(free_q)[0]
            original_solve=model.solve
            report['free_x_diagnostic']=True
            model.solve=lambda joints,delta,lower,upper:(free_q,free_position,orientation)
            try: command(float(free_position[2]))
            finally: model.solve=original_solve
            report['free_x_measured']=measured()
            # Return through the measured fixed-horizontal pose with the same native controller.
            restore_q=np.array(results['guarded_fixed_x']['joints'])
            restore_q[1]+=.003
            restore_q[2]+=.003
            restore_position=model.pose(restore_q)[0]
            model.solve=lambda joints,delta,lower,upper:(restore_q,restore_position,orientation)
            try: command(float(restore_position[2]))
            finally: model.solve=original_solve
            command(float(restore_position[2]+.03))
            report['passed']=True
        except Exception as error:
            report['error']=str(error)
            raise
        finally:
            save()
    web.run_esi_control=verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
