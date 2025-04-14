import cv2

def display_instruction(frame, player_bbox, activation_line, juggle_count):
    """ Draws the activation line, bounding box fill, and instruction text """

    if player_bbox is not None and activation_line is not None:
        lower_bound_y = player_bbox[3]  # Player's lower bounding box
        knee_y = activation_line.start.y  # Activation line at knee level

        # Fill the area between activation line and lower bound with red
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(player_bbox[0]), int(knee_y)),
                      (int(player_bbox[2]), int(lower_bound_y)), (0, 0, 255), -1)
        alpha = 0.3  # Transparency level
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw activation line
        cv2.line(frame, (int(activation_line.start.x), int(knee_y)),
                 (int(activation_line.end.x), int(knee_y)), (255, 0, 0), 2)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (55, 10),(400, 80), (0, 0, 255), -1)
    alpha = 0.3  # Transparency level
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # Instruction message
    cv2.putText(frame, "Make sure the ball exceeds", (60, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, "the activation line at the knee", (60, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
