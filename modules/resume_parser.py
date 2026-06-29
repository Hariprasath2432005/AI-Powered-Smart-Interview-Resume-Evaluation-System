SKILLS_DB = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "machine learning",
    "deep learning",
    "nlp",
    "data science",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "streamlit",
    "excel",
    "power bi"
]

def extract_skills(text):   #This function finds skills from the resume text.

    text = text.lower()   #text = text.lower()
    found_skills = []   #This stores detected skills.

    for skill in SKILLS_DB:    #Loop through skill database.
        if skill in text:
            found_skills.append(skill)

    return found_skills