import sqlite3

# Connect to your database
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Extract title, content, formulas, and tags from your topics table
cursor.execute("SELECT title, content, formulas, tags FROM topics")
rows = cursor.fetchall()

# Write into a clean text training file
with open("training_data.txt", "w", encoding="utf-8") as f:
  for row in rows:
    title, content, formulas, tags = row
    text = (
        f"Topic: {title}\nContent:"
        f" {content}\nFormula/Equation: {formulas}\nTags: {tags}\n\n"
    )
    f.write(text)

conn.close()
print("Successfully extracted database content to 'training_data.txt'!")