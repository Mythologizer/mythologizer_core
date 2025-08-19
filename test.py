# Modern way to access schemas
from mythologizer_postgres import list_schema_files, get_schema_content

# List all available schema files
schema_files = list_schema_files()
print("Available schema files:", schema_files)

# Get content of a specific schema file
try:
    content = get_schema_content("myths.sql.j2")
    print("Myths schema content:", content[:200] + "...")  # First 200 chars
except FileNotFoundError as e:
    print(f"Schema file not found: {e}")