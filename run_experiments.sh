#!/bin/bash

# Define ranges for parameters
SEGMENT_LENGTHS=(1.0 2.0 4.0 8.0 16.0 32.0)
THRESHOLDS=(0.6)
NEGATIVE_SAMPLES=20

# Create a results directory
RESULTS_DIR="experiment_results"
mkdir -p $RESULTS_DIR

# Log file for summary
SUMMARY_LOG="$RESULTS_DIR/summary.txt"
echo "Experiment Summary" > $SUMMARY_LOG
echo "=================" >> $SUMMARY_LOG
echo "Date: $(date)" >> $SUMMARY_LOG
echo "" >> $SUMMARY_LOG
echo "Parameters tested:" >> $SUMMARY_LOG
echo "Segment Lengths: ${SEGMENT_LENGTHS[*]}" >> $SUMMARY_LOG
echo "Thresholds: ${THRESHOLDS[*]}" >> $SUMMARY_LOG
echo "Negative Samples: $NEGATIVE_SAMPLES" >> $SUMMARY_LOG
echo "" >> $SUMMARY_LOG
echo "Results:" >> $SUMMARY_LOG
echo "-------" >> $SUMMARY_LOG

# Run experiments for each combination
for seg_len in "${SEGMENT_LENGTHS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
        echo "================================="
        echo "Running with segment length: $seg_len, threshold: $threshold"
        echo "================================="
        
        # Create a log file for this run
        LOG_FILE="$RESULTS_DIR/segment${seg_len}_threshold${threshold}.log"
        
        # Run the main.py script with these parameters
        python main.py --segment-length $seg_len --threshold $threshold --negative-samples $NEGATIVE_SAMPLES | tee $LOG_FILE
        
        # Extract key metrics for the summary
        POS_SUCCESS=$(grep "Positive Tests" -A 3 $LOG_FILE | grep "Success rate" | awk '{print $4}')
        NEG_SUCCESS=$(grep "Negative Tests" -A 3 $LOG_FILE | grep "Success rate" | awk '{print $4}')
        TOO_SHORT=$(grep "Too short:" $LOG_FILE | awk '{print $4}')
        
        # Add to summary
        echo "Segment Length: $seg_len, Threshold: $threshold" >> $SUMMARY_LOG
        echo "  - Too short files: $TOO_SHORT" >> $SUMMARY_LOG
        echo "  - Positive test success rate: $POS_SUCCESS" >> $SUMMARY_LOG
        echo "  - Negative test success rate: $NEG_SUCCESS" >> $SUMMARY_LOG
        echo "" >> $SUMMARY_LOG
    done
done

echo "==================================="
echo "All experiments completed!"
echo "Summary saved to $SUMMARY_LOG"
echo "Detailed logs saved in $RESULTS_DIR"
echo "==================================="

# Show summary
cat $SUMMARY_LOG 