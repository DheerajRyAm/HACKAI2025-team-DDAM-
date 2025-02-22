#!/bin/python3

high_score=0


emotion=["angry","disgust","fear","happy","neutral","sad","suprise"]
gesture=["plam","I","fist","fist_moved","thumb","index","ok","plam_moved","c","down"]

current_score=0
num_correct=0
while song_is_playing:
    
    random_interval_for_emotion -> then set to random_emotion
    random_interval_for_gesture -> then set to random_gesture

    # check if player got the correct emotion and gesture
    if emotion_value_from_ai == random_emotion:
        num_correct+=1
    else:
        num_correct=0

    if gesture_value_from_ai == random_gesture:
        num_correct+=1
    else:
        num_correct=0


    # combos!!!
    if num_correct <= 3:
        current_score = current_score + 5
    elif num_correct > 3 and num_correct < 7:
        current_score = current_score * 5
    elif num_correct > 7 and num_correct < 10:
        current_score = current_score * 10
    elif num_correct > 10 and num_correct < 20:
        current_score = current_score * 25
    elif num_correct > 20:
        current_score = current_score * 100
    
    # check if current score is better than high score
    if current_score > high_score:
        high_score = current_score


