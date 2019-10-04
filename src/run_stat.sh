#!/bin/bash

rm ../build/summary.txt

# make Task MP.7
../build/2D_feature_tracking MP7

# make Task MP.8
../build/2D_feature_tracking MP8 BRISK
../build/2D_feature_tracking MP8 BRIEF
../build/2D_feature_tracking MP8 ORB
../build/2D_feature_tracking MP8 FREAK
../build/2D_feature_tracking MP8 AKAZE
../build/2D_feature_tracking MP8 SIFT

# make Task MP.9
../build/2D_feature_tracking MP9 BRISK
../build/2D_feature_tracking MP9 BRIEF
../build/2D_feature_tracking MP9 ORB
../build/2D_feature_tracking MP9 FREAK
../build/2D_feature_tracking MP9 AKAZE
../build/2D_feature_tracking MP9 SIFT

