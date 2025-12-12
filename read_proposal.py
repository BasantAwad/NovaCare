from pypdf import PdfReader

try:
    reader = PdfReader("grad_project___proposal.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open("proposal_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Text extraction complete.")
except Exception as e:
    print(f"Error reading PDF: {e}")
