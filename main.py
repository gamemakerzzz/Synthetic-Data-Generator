

import logging
from src.utils.database import get_database_config, test_database_connection
from src.core.schema_analyzer import analyze_database_schema, export_analysis_config
from src.core.data_generator import DataGenerator
from src.core.validator import DataValidator, ValidationLevel, export_validation_report
from src.core.exporter import export_generated_data

def main():
    """
    Main function to orchestrate the data generation, validation, and export process.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 50)
    print("Step 1: Database Configuration")
    print("=" * 50)
    db_config = get_database_config()
    if not test_database_connection(db_config):
        print("Database connection failed. Exiting.")
        return

    print("\n" + "=" * 50)
    print("Step 2: Analyzing Database Schema")
    print("=" * 50)
    try:
        analysis_result = analyze_database_schema(db_config)
        print(f"Analysis complete. Found {len(analysis_result.tables)} tables.")
        export_analysis_config(analysis_result, 'src/output/schema_analysis.json')
        print("Schema analysis saved to src/output/schema_analysis.json")
    except Exception as e:
        logging.error(f"Schema analysis failed: {e}")
        return

    print("\n" + "=" * 50)
    print("Step 3: Generating Data")
    print("=" * 50)
    try:
        records_input = input("Enter number of records per table (default 100): ").strip()
        records_per_table = int(records_input) if records_input.isdigit() else 100
        locale = input(f"Enter locale {analysis_result.suggested_locales} (default en_US): ").strip() or 'en_US'
        
        generator = DataGenerator(config=db_config, locale=locale)
        generated_data = generator.generate_schema_data(analysis_result, records_per_table)
        summary = generator.get_generation_summary()
        print(f"\nGeneration complete! Total records: {summary['total_records']}")
    except Exception as e:
        logging.error(f"Data generation failed: {e}")
        return

    print("\n" + "=" * 50)
    print("Step 4: Validating Generated Data")
    print("=" * 50)
    try:
        print("Select validation level:")
        print("1) Strict (all checks, including relations)")
        print("2) Standard (type, format, and business rule checks)")
        print("3) Lenient (basic checks)")
        level_choice = input("Enter choice (default 2): ").strip() or '2'
        level_map = {'1': ValidationLevel.STRICT, '2': ValidationLevel.STANDARD, '3': ValidationLevel.LENIENT}
        validation_level = level_map.get(level_choice, ValidationLevel.STANDARD)

        validator = DataValidator(analysis_result, validation_level=validation_level, locale=locale)
        report = validator.validate_generated_data(generated_data)

        print(f"\nValidation Complete. Overall Quality Score: {report.quality_score:.2%}")
        print(f"Total Issues Found: {report.total_issues}")
        if report.total_issues > 0:
            print("Issue Summary:")
            for issue_type, count in report.issues_by_type.items():
                print(f"- {issue_type}: {count}")

        # Export the validation report
        export_validation_report(report)

        quality_threshold = 0.75
        if report.quality_score < quality_threshold:
            print(f"\nValidation FAILED: Quality score ({report.quality_score:.2%}) is below the required threshold ({quality_threshold:.2%}).")
            print("Halting process. Please review the detailed issues and regenerate the data.")
            return
        else:
            print(f"\nValidation PASSED: Quality score is acceptable.")

    except Exception as e:
        logging.error(f"Data validation failed: {e}")
        return

    print("\n" + "=" * 50)
    print("Step 5: Exporting Data")
    print("=" * 50)
    try:
        print("Choose an export option:")
        print("1) Export to files (JSON, CSV, Excel, etc.)")
        print("2) Insert into database")
        choice = input("\nEnter your choice (1/2): ").strip()

        if choice == '1':
            print("\nAvailable file formats: json, csv, excel, parquet, jsonl")
            formats_input = input("Enter desired formats (comma-separated, e.g., json,csv): ").strip()
            formats = [f.strip() for f in formats_input.split(',')] if formats_input else ['json']
            single_file = input("Export as a single file? (y/n): ").strip().lower() == 'y'
            export_result = export_generated_data(
                generated_data,
                export_type='file',
                formats=formats,
                single_file=single_file
            )
        elif choice == '2':
            export_result = export_generated_data(
                generated_data, 
                export_type='database',
                db_config=db_config
            )
        else:
            print("Invalid choice. Exiting.")
            return

        print(f"\nExport complete: {export_result.message}")
        if export_result.errors:
            print("Export finished with some errors:")
            for error in export_result.errors:
                print(f" - {error}")

    except Exception as e:
        logging.error(f"Data export failed: {e}")

if __name__ == "__main__":
    main()
