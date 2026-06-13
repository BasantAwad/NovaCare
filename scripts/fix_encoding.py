import os

REPLACEMENTS = {
    'Â·': '·',
    'â€”': '—',
    'â€¢': '•',
    'Â°': '°',
    'â€¦': '…',
    'â†’': '→',
    'Â§': '§',
    'â”€': '─',
    'â•': '═'
}

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for k, v in REPLACEMENTS.items():
        content = content.replace(k, v)
        
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")

def main():
    root = r"c:\Users\Pc\NovaCare-1\novacare_app\lib"
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".dart"):
                fix_file(os.path.join(dirpath, filename))

if __name__ == "__main__":
    main()
