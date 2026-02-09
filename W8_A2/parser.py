{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2cf51004-504e-4f9d-a690-7a648f0d9927",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pypdf\n",
    "from io import BytesIO\n",
    "\n",
    "class PDFParser:\n",
    "    \"\"\"Handles parsing of PDF files to extract text.\"\"\"\n",
    "    \n",
    "    def extract_text(self, file_path: str) -> str:\n",
    "        \"\"\"\n",
    "        Reads a PDF file and returns its textual content.\n",
    "        \"\"\"\n",
    "        try:\n",
    "            text_content = []\n",
    "            with open(file_path, 'rb') as f:\n",
    "                reader = pypdf.PdfReader(f)\n",
    "                for page in reader.pages:\n",
    "                    text_content.append(page.extract_text())\n",
    "            \n",
    "            full_text = \"\\n\".join(text_content)\n",
    "            if not full_text.strip():\n",
    "                raise ValueError(\"No text found in PDF. It might be an image scan.\")\n",
    "            return full_text\n",
    "            \n",
    "        except Exception as e:\n",
    "            raise RuntimeError(f\"Failed to parse PDF: {e}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
