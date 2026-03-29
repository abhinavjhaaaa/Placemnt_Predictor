from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import re
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)

# ==============================
# 🔹 LOAD MODEL (with error handling)
# ==============================
try:
    model = joblib.load("models/btech_cse_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    model = None
    scaler = None


# ==============================
# 🔹 PLACEMENT SUGGESTION
# ==============================
def generate_suggestion(data):
    suggestions = []

    if data.get("cgpa", 0) < 7:
        suggestions.append("Improve your CGPA (target 7.5+)")
    if data.get("dsa", 0) < 3:
        suggestions.append("Strengthen DSA skills (practice on LeetCode/GFG)")
    if data.get("projects", 0) < 2:
        suggestions.append("Build 2-3 quality projects")
    if data.get("internship", 0) == 0:
        suggestions.append("Gain internship experience")
    if data.get("communication", 0) < 3:
        suggestions.append("Improve communication skills")
    if data.get("aptitude", 0) < 60:
        suggestions.append("Practice aptitude tests regularly")
    if data.get("certifications", 0) < 2:
        suggestions.append("Add relevant certifications")
    if data.get("consistency", 0) < 6:
        suggestions.append("Maintain consistent study habits")
    if data.get("score", 0) < 70:
        suggestions.append("Improve academic score")

    if not suggestions:
        return "Excellent profile! You are on track for placement 🚀"

    return "Focus on: " + ", ".join(suggestions[:3])


# ==============================
# 🔹 HOME ROUTE
# ==============================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Placement Prediction API is running",
        "endpoints": ["/predict", "/analyze_resume", "/upload_resume"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# ==============================
# 🔹 PLACEMENT PREDICTION
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['cgpa', 'dsa', 'projects', 'internship', 
                          'communication', 'aptitude', 'certifications', 
                          'consistency', 'score']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare features
        features = np.array([[
            float(data["cgpa"]),
            int(data["dsa"]),
            int(data["projects"]),
            int(data["internship"]),
            int(data["communication"]),
            float(data["aptitude"]),
            int(data["certifications"]),
            float(data["consistency"]),
            float(data["score"])
        ]])
        
        # Scale features if model exists
        if scaler is not None:
            features = scaler.transform(features)
        
        # Predict
        if model is not None:
            raw_prob = model.predict_proba(features)[0][1]
            probability = 0.2 + (raw_prob * 0.6)
            probability = min(max(probability, 0.1), 0.95)
            prediction = 1 if probability >= 0.6 else 0
        else:
            # Fallback logic if model not loaded
            probability = 0.2 + (
                float(data["cgpa"]) / 10 * 0.2 +
                int(data["dsa"]) / 5 * 0.15 +
                int(data["projects"]) / 5 * 0.1 +
                int(data["internship"]) * 0.1 +
                int(data["communication"]) / 5 * 0.1 +
                float(data["aptitude"]) / 100 * 0.15 +
                int(data["certifications"]) / 5 * 0.05 +
                float(data["consistency"]) / 10 * 0.1 +
                float(data["score"]) / 100 * 0.05
            )
            probability = min(max(probability, 0.1), 0.9)
            prediction = 1 if probability >= 0.6 else 0
        
        probability = round(probability * 100) / 100
        
        # Confidence level
        if probability >= 0.75:
            confidence = "High"
        elif probability >= 0.45:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        suggestion = generate_suggestion(data)
        
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": confidence,
            "suggestion": suggestion
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔥 PROFESSIONAL RESUME ANALYZER WITH STRICT SCORING
# ==============================
def process_resume_text(resume_text):
    # ========== VALIDATION: Check for meaningful content ==========
    if not resume_text or resume_text.strip() == "":
        return {
            "score": 0,
            "skills": [],
            "missing_skills": ["Complete resume content"],
            "suggestions": "Please provide your resume content for analysis.",
            "confidence": "Low"
        }
    
    # Check if resume has meaningful content (not just gibberish)
    resume_text_lower = resume_text.lower().strip()
    
    # Remove whitespace and check if it's just a single character
    if len(resume_text_lower) < 50:
        return {
            "score": 0,
            "skills": [],
            "missing_skills": ["Complete resume content", "Education", "Skills", "Projects", "Experience"],
            "suggestions": "Your resume is too short. Please add detailed information about your education, skills, projects, and experience.",
            "confidence": "Low"
        }
    
    # Check for common resume sections (must have at least 2)
    has_education = any(word in resume_text_lower for word in ["education", "b.tech", "bachelor", "degree", "university", "college", "institute", "school"])
    has_skills = any(word in resume_text_lower for word in ["skills", "technologies", "technical skills", "tech stack", "programming languages"])
    has_projects = "project" in resume_text_lower
    has_experience = any(word in resume_text_lower for word in ["experience", "internship", "work experience", "employment", "worked at"])
    
    # Minimum requirement: Must have at least education OR skills
    if not has_education and not has_skills:
        return {
            "score": 0,
            "skills": [],
            "missing_skills": ["Education details", "Technical skills"],
            "suggestions": "Your resume is missing essential sections. Please add your education details and technical skills.",
            "confidence": "Low"
        }
    
    # ========== SKILLS DATABASE ==========
    TECH_SKILLS = {
        "programming": ["python", "java", "c++", "c", "javascript", "typescript", "go", "rust", "ruby", "php", "swift", "kotlin", "scala"],
        "web": ["react", "angular", "vue", "node", "node.js", "express", "django", "flask", "spring", "spring boot", "html", "css", "tailwind"],
        "database": ["sql", "mysql", "postgresql", "mongodb", "oracle", "redis", "firebase", "dynamodb"],
        "cloud_devops": ["aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins", "git", "github", "ci/cd", "terraform"],
        "data_ml": ["pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "machine learning", "deep learning", "nlp", "data science", "ai"]
    }
    
    SOFT_SKILLS = [
        "communication", "leadership", "teamwork", "problem solving", "critical thinking",
        "adaptability", "time management", "collaboration", "analytical", "creativity",
        "project management", "agile", "scrum", "presentation", "interpersonal"
    ]
    
    # Flatten skills for detection
    all_tech_skills = []
    for category in TECH_SKILLS.values():
        all_tech_skills.extend(category)
    
    # Find skills (exact matches)
    found_tech = list(set([s for s in all_tech_skills if s in resume_text_lower]))
    found_soft = list(set([s for s in SOFT_SKILLS if s in resume_text_lower]))
    
    # ========== SECTION DETECTION ==========
    sections = {
        "education": has_education,
        "projects": has_projects,
        "experience": has_experience,
        "skills": has_skills,
        "certifications": any(word in resume_text_lower for word in ["certification", "certified", "certificate", "course", "credential"]),
        "achievements": any(word in resume_text_lower for word in ["achievement", "award", "recognition", "honor", "winner", "scholarship"])
    }
    
    # ========== WORD COUNT (meaningful content only) ==========
    # Remove common headers and contact info
    lines = resume_text_lower.split('\n')
    meaningful_lines = []
    skip_headers = ["email:", "phone:", "github:", "linkedin:", "address:", "contact:"]
    
    for line in lines:
        if not any(header in line for header in skip_headers) and len(line.strip()) > 5:
            meaningful_lines.append(line)
    
    word_count = len(' '.join(meaningful_lines).split())
    
    # ========== PROFESSIONAL SCORING SYSTEM ==========
    score_components = {}
    
    # 1. Technical Skills Score (max 30 points)
    tech_skill_count = len(found_tech)
    if tech_skill_count >= 12:
        skill_score = 30
    elif tech_skill_count >= 10:
        skill_score = 27
    elif tech_skill_count >= 8:
        skill_score = 24
    elif tech_skill_count >= 6:
        skill_score = 20
    elif tech_skill_count >= 4:
        skill_score = 15
    elif tech_skill_count >= 2:
        skill_score = 8
    elif tech_skill_count >= 1:
        skill_score = 3
    else:
        skill_score = 0
    
    # Soft skills bonus (max 5)
    if found_soft:
        soft_bonus = min(len(found_soft) * 2, 5)
        skill_score += soft_bonus
    
    skill_score = min(skill_score, 30)
    score_components["technical_skills"] = skill_score
    
    # 2. Section Coverage Score (max 35 points)
    required_sections = ["education", "projects", "experience", "skills"]
    optional_sections = ["certifications", "achievements"]
    
    required_count = sum(1 for sec in required_sections if sections[sec])
    optional_count = sum(1 for sec in optional_sections if sections[sec])
    
    # Required sections are mandatory
    if required_count == 4:
        section_score = 30
    elif required_count == 3:
        section_score = 22
    elif required_count == 2:
        section_score = 14
    elif required_count == 1:
        section_score = 6
    else:
        section_score = 0
    
    # Bonus for optional sections
    section_score += optional_count * 3
    section_score = min(section_score, 35)
    score_components["section_coverage"] = section_score
    
    # 3. Content Quality Score (max 20 points)
    quality_score = 0
    
    # Check for quantifiable achievements (numbers with %)
    has_percentages = bool(re.search(r'\d+%', resume_text_lower))
    if has_percentages:
        quality_score += 6
    
    # Check for metrics (numbers)
    has_metrics = bool(re.search(r'\d+\s*(?:users|customers|clients|revenue|sales|performance|lines|reduced|increased|improved)', resume_text_lower))
    if has_metrics:
        quality_score += 4
    
    # Check for project links
    if "github.com" in resume_text_lower or "gitlab.com" in resume_text_lower or "portfolio" in resume_text_lower:
        quality_score += 3
    
    # Check for action verbs
    action_words = ["developed", "built", "designed", "implemented", "led", "created", "managed", "spearheaded", "architected", "optimized", "improved", "achieved"]
    action_count = sum([1 for word in action_words if word in resume_text_lower])
    quality_score += min(action_count, 7)
    
    quality_score = min(quality_score, 20)
    score_components["content_quality"] = quality_score
    
    # 4. Resume Length Score (max 15 points)
    if word_count >= 500:
        length_score = 15
    elif word_count >= 400:
        length_score = 13
    elif word_count >= 350:
        length_score = 11
    elif word_count >= 300:
        length_score = 9
    elif word_count >= 250:
        length_score = 7
    elif word_count >= 200:
        length_score = 5
    elif word_count >= 150:
        length_score = 3
    elif word_count >= 100:
        length_score = 1
    else:
        length_score = 0
    score_components["resume_length"] = length_score
    
    # Calculate total score
    total_score = sum(score_components.values())
    
    # ========== DETERMINE MISSING SKILLS ==========
    # Important industry skills
    important_skills = ["python", "java", "javascript", "sql", "react", "aws", "docker", "git", "node.js", "mongodb"]
    missing_important = [s for s in important_skills if s not in found_tech]
    
    # Missing critical sections
    missing_critical = []
    if not sections["education"]:
        missing_critical.append("Education details")
    if not sections["projects"]:
        missing_critical.append("Projects section")
    if not sections["experience"]:
        missing_critical.append("Experience/Internship")
    if not sections["skills"]:
        missing_critical.append("Skills section")
    
    # Combine missing items
    all_missing = missing_critical + missing_important[:5]
    
    # ========== GENERATE SUGGESTIONS ==========
    suggestions = []
    
    # Critical missing sections (highest priority)
    if missing_critical:
        suggestions.append(f"CRITICAL: Add missing sections: {', '.join(missing_critical)}")
    
    # Skills improvement
    if tech_skill_count < 6:
        suggestions.append(f"Add more technical skills (currently {tech_skill_count}, aim for 8+ industry-relevant skills)")
    
    if missing_important:
        suggestions.append(f"Learn in-demand skills: {', '.join(missing_important[:4])}")
    
    # Content quality
    if not has_percentages and not has_metrics:
        suggestions.append("Add quantifiable achievements (e.g., 'Improved performance by 30%', 'Reduced load time by 50%')")
    
    if action_count < 5:
        suggestions.append("Use strong action verbs (developed, implemented, optimized, led, architected)")
    
    # Length improvement
    if word_count < 300:
        suggestions.append(f"Expand your resume (currently {word_count} words, aim for 350-500 words)")
    elif word_count > 700:
        suggestions.append("Consider condensing to 1-2 pages (aim for 400-600 words)")
    
    # Positive feedback for good resumes
    if total_score >= 80 and not suggestions:
        suggestions.append("Excellent resume! Consider tailoring for specific job applications and adding more metrics")
    elif total_score >= 70:
        suggestions.append("Good resume! Focus on adding more quantifiable achievements")
    
    # Ensure we have at least one suggestion
    if not suggestions:
        suggestions.append("Review your resume for consistency and consider adding more specific metrics")
    
    # ========== CONFIDENCE LEVEL ==========
    if total_score >= 85:
        confidence = "High"
    elif total_score >= 70:
        confidence = "Medium-High"
    elif total_score >= 55:
        confidence = "Medium"
    elif total_score >= 35:
        confidence = "Medium-Low"
    else:
        confidence = "Low"
    
    return {
        "score": total_score,
        "skills": found_tech[:20],
        "missing_skills": all_missing[:8],
        "suggestions": " ".join(suggestions[:5]),
        "confidence": confidence,
        "word_count": word_count,
        "sections_found": [k for k, v in sections.items() if v],
        "score_breakdown": score_components
    }


# ==============================
# 🔹 TEXT RESUME ANALYZER
# ==============================
@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    try:
        data = request.json
        
        # Handle both "resume" and "text" field names
        resume_text = data.get("resume") or data.get("text", "")
        
        if not resume_text:
            return jsonify({
                "score": 0,
                "skills": [],
                "missing_skills": ["No text provided"],
                "suggestions": "Please provide your resume content for analysis.",
                "confidence": "Low"
            }), 400
        
        result = process_resume_text(resume_text)
        return jsonify(result)
        
    except Exception as e:
        print(f"Resume analysis error: {e}")
        return jsonify({
            "score": 0,
            "skills": [],
            "missing_skills": ["Error processing resume"],
            "suggestions": f"Error analyzing resume. Please check the format.",
            "confidence": "Low"
        }), 500


# ==============================
# 🔹 PDF RESUME ANALYZER
# ==============================
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    try:
        if "file" not in request.files:
            return jsonify({
                "score": 0,
                "skills": [],
                "missing_skills": ["No file uploaded"],
                "suggestions": "Please upload a PDF file containing your resume.",
                "confidence": "Low"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "score": 0,
                "skills": [],
                "missing_skills": ["Empty file"],
                "suggestions": "Please select a valid PDF file.",
                "confidence": "Low"
            }), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                "score": 0,
                "skills": [],
                "missing_skills": ["Invalid file type"],
                "suggestions": "Only PDF files are accepted. Please convert your resume to PDF.",
                "confidence": "Low"
            }), 400

        # Read PDF
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text.strip():
            return jsonify({
                "score": 0,
                "skills": [],
                "missing_skills": ["No text extracted"],
                "suggestions": "Could not extract text from PDF. Ensure it's not an image-based PDF and contains selectable text.",
                "confidence": "Low"
            }), 400

        result = process_resume_text(text)
        return jsonify(result)

    except Exception as e:
        print(f"PDF upload error: {e}")
        return jsonify({
            "score": 0,
            "skills": [],
            "missing_skills": ["Error processing PDF"],
            "suggestions": f"Error reading PDF file. Please ensure it's a valid PDF document.",
            "confidence": "Low"
        }), 500


# ==============================
# 🔹 SUPPORTED STREAMS
# ==============================
@app.route("/streams", methods=["GET"])
def get_streams():
    return jsonify({
        "streams": [
            {"id": "btech", "name": "B.Tech", "specializations": ["cse"]}
        ]
    })


# ==============================
# 🔹 RUN APP
# ==============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 PlaceIQ Professional Backend Server")
    print("="*60)
    print("📍 Running at: http://127.0.0.1:5000")
    print("📋 Available Endpoints:")
    print("   POST /predict          - Placement prediction")
    print("   POST /analyze_resume   - Resume text analysis")
    print("   POST /upload_resume    - PDF resume upload")
    print("   GET  /health           - Health check")
    print("   GET  /streams          - Supported streams")
    print("="*60)
    print("\n✅ Scoring System:")
    print("   • Technical Skills: 30 points")
    print("   • Section Coverage: 35 points")
    print("   • Content Quality: 20 points")
    print("   • Resume Length: 15 points")
    print("   • Total: 100 points")
    print("\n💡 Minimum Requirements:")
    print("   • Must have education OR skills section")
    print("   • Minimum 50 characters")
    print("   • Proper resume structure")
    print("="*60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)