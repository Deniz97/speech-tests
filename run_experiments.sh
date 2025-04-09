#!/bin/bash

# Define ranges for parameters
SEGMENT_LENGTHS=(32.0 16.0 8.0 4.0 2.0 1.0)
THRESHOLDS=(0.6)
NEGATIVE_SAMPLES=10

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

# CSV file for results (easier for analysis)
CSV_FILE="$RESULTS_DIR/results.csv"
echo "SegmentLength,Threshold,TotalFiles,SuccessfulFiles,TooShortFiles,OtherFailedFiles,TotalPositiveTests,SuccessfulPositiveTests,PositiveSuccessRate,AvgPositiveSimilarity,FilesWithPositiveFailures,TotalNegativeTests,SuccessfulNegativeTests,NegativeSuccessRate,AvgNegativeSimilarity,FilesWithNegativeFailures" > $CSV_FILE

# Run experiments for each combination
for seg_len in "${SEGMENT_LENGTHS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
        echo "================================="
        echo "Running with segment length: $seg_len, threshold: $threshold"
        echo "================================="
        
        # Create a log file for this run
        LOG_FILE="$RESULTS_DIR/segment${seg_len}_threshold${threshold}.log"
        
        # Run the main.py script with these parameters
        python main.py --segment-length $seg_len --threshold $threshold --negative-samples $NEGATIVE_SAMPLES --customer-only --max-files 100 | tee $LOG_FILE
        
        # Extract key metrics for the summary
        TOTAL_FILES=$(grep "Total files processed:" $LOG_FILE | awk '{print $4}')
        SUCCESSFUL_FILES=$(grep "Successful:" $LOG_FILE | awk '{print $3}')
        TOO_SHORT=$(grep "Too short:" $LOG_FILE | awk '{print $4}')
        OTHER_FAILED=$(grep "Other failures:" $LOG_FILE | awk '{print $4}')
        
        # Extract positive test metrics
        TOTAL_POS_TESTS=$(grep "Positive Tests" -A 5 $LOG_FILE | grep "Total tests:" | awk '{print $4}')
        SUCCESSFUL_POS_TESTS=$(grep "Positive Tests" -A 5 $LOG_FILE | grep "Successful matches:" | awk '{print $4}')
        POS_SUCCESS_RATE=$(grep "Positive Tests" -A 5 $LOG_FILE | grep "Success rate:" | awk '{print $4}')
        AVG_POS_SIMILARITY=$(grep "Positive Tests" -A 5 $LOG_FILE | grep "Average similarity:" | awk '{print $4}')
        FILES_WITH_POS_FAILURES=$(grep "Files with at least one positive failure:" $LOG_FILE | awk '{print $NF}')
        
        # Extract negative test metrics
        TOTAL_NEG_TESTS=$(grep "Negative Tests" -A 5 $LOG_FILE | grep "Total tests:" | awk '{print $4}')
        SUCCESSFUL_NEG_TESTS=$(grep "Negative Tests" -A 5 $LOG_FILE | grep "Successful non-matches:" | awk '{print $4}')
        NEG_SUCCESS_RATE=$(grep "Negative Tests" -A 5 $LOG_FILE | grep "Success rate:" | awk '{print $4}')
        AVG_NEG_SIMILARITY=$(grep "Negative Tests" -A 5 $LOG_FILE | grep "Average similarity:" | awk '{print $4}')
        FILES_WITH_NEG_FAILURES=$(grep "Files with at least one negative failure:" $LOG_FILE | awk '{print $NF}')
        
        # Add to summary
        echo "Segment Length: $seg_len, Threshold: $threshold" >> $SUMMARY_LOG
        echo "Timestamp: $(date +"%Y-%m-%d %H:%M:%S")" >> $SUMMARY_LOG
        echo "  - Total files processed: $TOTAL_FILES" >> $SUMMARY_LOG
        echo "  - Successful files: $SUCCESSFUL_FILES" >> $SUMMARY_LOG
        echo "  - Too short files: $TOO_SHORT" >> $SUMMARY_LOG
        echo "  - Other failed files: $OTHER_FAILED" >> $SUMMARY_LOG
        echo "" >> $SUMMARY_LOG
        echo "  Positive Tests:" >> $SUMMARY_LOG
        echo "  - Total tests: $TOTAL_POS_TESTS" >> $SUMMARY_LOG
        echo "  - Successful matches: $SUCCESSFUL_POS_TESTS" >> $SUMMARY_LOG
        echo "  - Success rate: $POS_SUCCESS_RATE" >> $SUMMARY_LOG
        echo "  - Average similarity: $AVG_POS_SIMILARITY" >> $SUMMARY_LOG
        echo "  - Files with positive failures: $FILES_WITH_POS_FAILURES" >> $SUMMARY_LOG
        echo "" >> $SUMMARY_LOG
        echo "  Negative Tests:" >> $SUMMARY_LOG
        echo "  - Total tests: $TOTAL_NEG_TESTS" >> $SUMMARY_LOG
        echo "  - Successful non-matches: $SUCCESSFUL_NEG_TESTS" >> $SUMMARY_LOG
        echo "  - Success rate: $NEG_SUCCESS_RATE" >> $SUMMARY_LOG
        echo "  - Average similarity: $AVG_NEG_SIMILARITY" >> $SUMMARY_LOG
        echo "  - Files with negative failures: $FILES_WITH_NEG_FAILURES" >> $SUMMARY_LOG
        echo "" >> $SUMMARY_LOG
        
        # Add to CSV file (without % sign for easier processing)
        POS_RATE_NUM=${POS_SUCCESS_RATE/\%/}
        NEG_RATE_NUM=${NEG_SUCCESS_RATE/\%/}
        echo "$seg_len,$threshold,$TOTAL_FILES,$SUCCESSFUL_FILES,$TOO_SHORT,$OTHER_FAILED,$TOTAL_POS_TESTS,$SUCCESSFUL_POS_TESTS,$POS_RATE_NUM,$AVG_POS_SIMILARITY,$FILES_WITH_POS_FAILURES,$TOTAL_NEG_TESTS,$SUCCESSFUL_NEG_TESTS,$NEG_RATE_NUM,$AVG_NEG_SIMILARITY,$FILES_WITH_NEG_FAILURES" >> $CSV_FILE
    done
done

echo "==================================="
echo "All experiments completed!"
echo "Summary saved to $SUMMARY_LOG"
echo "CSV data saved to $CSV_FILE"
echo "Detailed logs saved in $RESULTS_DIR"
echo "==================================="

# Show summary
cat $SUMMARY_LOG 