from factory import ExporterFactory

def main():
    # The dataset to export
    dataset = [
        {"id": 1, "name": "Alice Smith", "role": "Developer", "active": True},
        {"id": 2, "name": "Bob Jones", "role": "Designer", "active": False},
        {"id": 3, "name": "Charlie Day", "role": "Manager", "active": True},
    ]

    print("Available formats: CSV, JSON, XML")
    choice = input("Enter desired export format: ")

    try:
        # 1. Ask the Factory for the correct exporter object
        exporter = ExporterFactory.get_exporter(choice)
        
        # 2. Use the exporter (Client doesn't care which one it is)
        filename = f"export_data.{choice.lower()}"
        exporter.export(dataset, filename)
        
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()