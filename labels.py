#
# File: labels.py
# Description: defines the mapping between labels and numerical class codes.
# Author: Atis Elsts, 2019
#

LABEL_TO_CODE = {
    "UNKNOWN" : -1,
    "WALKING" : 1,
    "WALKING_UPSTAIRS" : 2,
    "WALKING_DOWNSTAIRS" : 3,
    "SITTING" : 4,
    "STANDING" : 5,
    "LAYING" : 6,
    "a_ascend" : 7,
    "a_descend" : 8,
    "a_jump" : 9,
    "p_bent" : 10,
    "p_kneel" : 11,
    "p_squat" : 12,
    "t_bend" : 13,
    "t_kneel_stand" : 14,
    "t_lie_sit" : 15,
    "t_sit_lie" : 16,
    "t_sit_stand" : 17,
    "t_stand_kneel" : 18,
    "t_stand_sit" : 19,
    "t_straighten" : 20,
    "t_turn" : 21,
}

CODE_TO_LABEL = {
    -1 : "UNKNOWN",
    1 : "WALKING",
    2 : "WALKING_UPSTAIRS",
    3 : "WALKING_DOWNSTAIRS",
    4 : "SITTING",
    5 : "STANDING",
    6 : "LAYING",
    7 : "a_ascend",
    8 : "a_descend",
    9 : "a_jump",
    10 : "p_bent",
    11 : "p_kneel",
    12 : "p_squat",
    13 : "t_bend",
    14 : "t_kneel_stand",
    15 : "t_lie_sit",
    16 : "t_sit_lie",
    17 : "t_sit_stand",
    18 : "t_stand_kneel",
    19 : "t_stand_sit",
    20 : "t_straighten",
    21 : "t_turn",
}

ACTIVITY_SYNONYMS = {
    "a_walk" : "WALKING",
    "p_stand" : "STANDING",
    "p_sit" : "SITTING",
    "p_lie" : "LAYING",
    "WALKING" : "a_walk",
    "STANDING" : "p_stand",
    "SITTING" : "p_sit",
    "LAYING" : "p_lie",

}
