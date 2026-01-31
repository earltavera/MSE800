
from exporters import CsvExporter, JsonExporter, XmlExporter

class ExporterFactory:
    @staticmethod
    def get_exporter(format_type: str):
        format_type = format_type.lower().strip()
        
        if format_type == "csv":
            return CsvExporter()
        elif format_type == "json":
            return JsonExporter()
        elif format_type == "xml":
            return XmlExporter()
        else:
            raise ValueError(f"Unknown format: {format_type}")