TRAIN_CMD1="python3 model1.py npu:0"
TRAIN_CMD2="python3 model2.py npu:0"
TRAIN_CMD3="python3 model3.py cpu"
TRAIN_CMD4="python3 model4.py cpu"

$TRAIN_CMD1 &
PID1=$!

$TRAIN_CMD2 &
PID2=$!

$TRAIN_CMD3 &
PID3=$!

$TRAIN_CMD4 &
PID4=$!


wait $PID1
wait $PID2
wait $PID3
wait $PID4

echo "All training processes have completed."
