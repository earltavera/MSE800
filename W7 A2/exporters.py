import csv
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

# 1. The Abstract Interface
class DataExporter(ABC):
    @abstractmethod
    def export(self, data: list, filename: str):
        pass

# 2. Concrete Implementation: CSV
class CsvExporter(DataExporter):
    def export(self, data: list, filename: str):
        if not data:
            return
        
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"✅ Data successfully exported to {filename}")

# 3. Concrete Implementation: JSON
class JsonExporter(DataExporter):
    def export(self, data: list, filename: str):
        with open(filename, mode='w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Data successfully exported to {filename}")

# 4. Concrete Implementation: XML
class XmlExporter(DataExporter):
    def export(self, data: list, filename: str):
        root = ET.Element("root")
        
        for item in data:
            record = ET.SubElement(root, "record")
            for key, value in item.items():
                child = ET.SubElement(record, key)
                child.text = str(value)
                
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        print(f"✅ Data successfully exported to {filename}")