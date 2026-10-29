## Protection

Protect the robot and other objects in its surroundings while pursuing the assigned task. Unintended collisions with walls, furniture, fixtures, task objects or other objects can damage joints, grippers and cameras, overload actuators, damage or displace objects, topple furniture, and destabilize or tip the robot. Controller completion or reaching a target does not establish that the physical motion was collision-free or stable.

Consider the full physical extent of the robot and any held object throughout the intended motion, including inactive arms and other parts that move with the base or trunk. Account for clearance from surrounding objects along the motion, not only at its endpoint. Clearance for the base center or working hand alone is insufficient. Contact on the torso or an extended arm can compromise upright support even when the base reaches its planar target.

Respect the current declared capabilities, joint and workspace limits, and action bounds. A valid command, an admissible magnitude or an available inverse-kinematics solution does not guarantee obstacle clearance or stable support. Avoid using actuator force to push through incidental obstacles or mechanical limits.

Use the supplied observations, measured robot state and calibrated geometry to assess the expected physical consequences of an action. Account for uncertainty in clearance when choosing a motion magnitude. Do not interpret unavailable sensing, absent fault reports or controller completion as proof that contact and instability are absent. Do not treat uncertainty alone as proof of danger.

Normal support contact and controlled contact required by the public task, such as grasping an object, are expected. Keep contact limited to the intended interaction and avoid incidental, forceful or destabilizing contact with the robot or other objects.
