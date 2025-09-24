from flask import Flask, render_template_string, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management

# Dummy credentials
USERNAME = "admin"
PASSWORD = "password"

# Simple HTML templates
login_page = """
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
  <h2>Login</h2>
  {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
  <form method="POST">
    Username: <input type="text" name="username"><br><br>
    Password: <input type="password" name="password"><br><br>
    <input type="submit" value="Login">
  </form>
</body>
</html>
"""

home_page = """
<!DOCTYPE html>
<html>
<head><title>Home</title></head>
<body>
  <h2>Welcome {{ session['user'] }}!</h2>
  <a href="{{ url_for('logout') }}">Logout</a>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid credentials. Try again."
    return render_template_string(login_page, error=error)

@app.route("/home")
def home():
    if "user" in session:
        return render_template_string(home_page, session=session)
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
