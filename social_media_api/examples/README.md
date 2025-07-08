# Examples

This directory contains examples demonstrating how to use the Social Media Video Generation API.

## Contents

1. **example_workflow.py** - Demonstrates a complete workflow from trend analysis to video generation
2. **error_handling_demo.py** - Shows how the system handles various error scenarios
3. **output/** - Contains example output files from the workflow
4. **error_logs/** - Contains example error logs

## Running the Examples

### Complete Workflow Example

This example demonstrates the entire pipeline from trend analysis to video generation:

```bash
python examples/example_workflow.py
```

The example will:
1. Analyze mock trend data
2. Perform NLP analysis
3. Generate a mock video
4. Create post text
5. Save all outputs to the `examples/output/` directory

### Error Handling Demo

This example demonstrates how the system handles various error scenarios:

```bash
python examples/error_handling_demo.py
```

The demo will simulate:
1. Empty trend data
2. Invalid trend data
3. API failure
4. Partial failure (video generated but post text failed)

Error logs will be saved to the `examples/error_logs/` directory.

## Example Output Files

- **trend_data.json** - Example trend data
- **nlp_analysis.json** - Example NLP analysis results
- **video_metadata.json** - Example video generation metadata
- **post_data.json** - Example post text generation results
- **final_report.json** - Example final report combining all results

## Using the Examples as Templates

You can use these examples as templates for your own implementations:

1. Copy the example file
2. Modify the parameters to match your requirements
3. Replace mock data with real data sources
4. Run the modified script

For more detailed information, refer to the main [API documentation](../README.md).