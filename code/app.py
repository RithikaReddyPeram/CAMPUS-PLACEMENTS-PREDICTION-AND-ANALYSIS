from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import sqlite3
import bcrypt  # For password hashing
import joblib  # For ML model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling

#  Get the absolute path of the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DB_PATH = os.path.join(BASE_DIR, "users.db")

print(f"Loading model from: {MODEL_PATH}")
print(f"Loading scaler from: {SCALER_PATH}")

#  Load Model and Scaler with error handling
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise SystemExit("Model or Scaler file not found. Please train the model first.")

# Connect to Database
def connect_db():
    return sqlite3.connect(DB_PATH)

# Create Users Table
with connect_db() as conn:
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()

@app.route('/')
def home():
    if 'user' in session:
        return render_template('index.html', username=session['user'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['user_name'].strip()
        password = request.form['password'].strip()

        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[0]):
            session['user'] = username  # Store login session
            return redirect(url_for('home'))
        else:
            flash("Invalid Username or Password", "error")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_username = request.form['user_name'].strip()
        new_password = request.form['password'].strip()

        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (new_username,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash("Username already exists!", "error")
            else:
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                               (new_username, hashed_password))
                conn.commit()
                flash("Registration successful! You can now log in.", "success")
                return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        features = [
            float(request.form[field]) for field in ['sslc', 'pu', 'ug', 'quants', 'logical', 'verbal', 'programming', 'communication', 'experience']
        ]

        # Hiring Logic
        if (
            features[0] >= 50 and features[1] >= 35 and features[2] >= 5.0 and
            features[3] >= 15 and features[4] >= 15 and features[5] >= 15 and
            features[6] >= 15 and features[7] >= 15 and features[8] >= 2
        ) or (
            features[0] >= 70 and features[1] >= 70 and features[2] >= 7.0 and
            features[3] >= 15 and features[4] >= 15 and features[5] >= 15 and
            features[6] >= 15 and features[7] >= 15 and features[8] >= 1
        ):
            result = "Congratulations! You are Hired 🎉"
        else:
            result = "Sorry, you are Not Hired. Keep improving! 💪"

        return render_template('index.html', prediction=result, username=session['user'])

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", username=session['user'])

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash("Your message has been submitted successfully!", "success")
        return redirect(url_for("contact"))
    return render_template("contactus.html")

if __name__ == '__main__':
    app.run(debug=True)
