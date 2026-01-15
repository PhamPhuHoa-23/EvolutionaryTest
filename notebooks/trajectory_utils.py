def reconstruct_trajectories(initial_pos, initial_vel, initial_heading, actions, dt=0.1):
    """
    Reconstruct trajectories using Unicycle Kinematic Model.
    initial_pos: [N, 2]
    initial_vel: [N, 2] or [N] (speed)
    initial_heading: [N] (radians)
    actions: [N, T, 2] where actions are (acceleration, steering_curvature)
    
    Returns:
        gen_positions: [N, T, 2]
        gen_velocities: [N, T, 2]
        gen_headings: [N, T]
    """
    N, T, _ = actions.shape
    
    # Initialize state
    curr_pos = torch.tensor(initial_pos, dtype=torch.float32)
    
    # Handle velocity: input can be vector or scalar speed
    if len(initial_vel.shape) == 2:
        curr_speed = torch.tensor(np.linalg.norm(initial_vel, axis=-1), dtype=torch.float32)
    else:
        curr_speed = torch.tensor(initial_vel, dtype=torch.float32)
        
    curr_heading = torch.tensor(initial_heading, dtype=torch.float32)
    
    # Outputs
    pos_list = []
    vel_list = []
    head_list = []
    
    # Simulation loop
    for t in range(T):
        # Parse actions (scaling matches TrafficGamer rollout.py)
        # action[0] is acceleration (scaled by 5)
        # action[1] is curvature/steering (scaled by 0.05)
        acc = torch.tensor(actions[:, t, 0], dtype=torch.float32).clip(-1, 1) * 5.0
        kappa = torch.tensor(actions[:, t, 1], dtype=torch.float32).clip(-1, 1) * 0.05
        
        # Basic Unicycle Model Update
        # 1. Update Position
        # dx = v * cos(h) * dt + 0.5 * a * cos(h) * dt^2
        # dy = v * sin(h) * dt + 0.5 * a * sin(h) * dt^2
        
        cos_h = torch.cos(curr_heading)
        sin_h = torch.sin(curr_heading)
        
        distance = curr_speed * dt + 0.5 * acc * (dt**2)
        
        next_pos = curr_pos.clone()
        next_pos[:, 0] += distance * cos_h
        next_pos[:, 1] += distance * sin_h
        
        # 2. Update Heading
        # h_new = h + kappa * distance
        next_heading = curr_heading + kappa * distance
        
        # 3. Update Speed
        # v_new = v + a * dt
        next_speed = curr_speed + acc * dt
        next_speed = torch.clamp(next_speed, min=0.0) # Speed shouldn't be negative
        
        # Store
        pos_list.append(next_pos)
        head_list.append(next_heading)
        
        # Compute velocity vector for output
        vel_vec = torch.stack([next_speed * torch.cos(next_heading), 
                               next_speed * torch.sin(next_heading)], dim=-1)
        vel_list.append(vel_vec)
        
        # Update state
        curr_pos = next_pos
        curr_heading = next_heading
        curr_speed = next_speed
        
    # Stack results [N, T, ...]
    gen_positions = torch.stack(pos_list, dim=1).numpy()
    gen_velocities = torch.stack(vel_list, dim=1).numpy()
    gen_headings = torch.stack(head_list, dim=1).numpy()
    
    return gen_positions, gen_velocities, gen_headings
