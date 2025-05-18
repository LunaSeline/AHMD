// Toggle theme
const themeToggle = document.getElementById("themeToggle");
const body = document.body;

// Load saved theme
if (localStorage.getItem("theme") === "dark") {
  body.classList.add("dark-mode");
  themeToggle.textContent = "🌞 Light Mode";
} else {
  themeToggle.textContent = "🌙 Dark Mode";
}

// Toggle handler
themeToggle.addEventListener("click", () => {
  body.classList.toggle("dark-mode");
  const isDark = body.classList.contains("dark-mode");
  themeToggle.textContent = isDark ? "🌞 Light Mode" : "🌙 Dark Mode";
  localStorage.setItem("theme", isDark ? "dark" : "light");
});

// Simulate listening
document.getElementById("listenBtn").addEventListener("click", () => {
  alert("Simulated listening... speech-to-text would go here.");
});

document.querySelector(".done-btn").addEventListener("click", () => {
  alert("Marked as done!");
});
