import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove Emojis
emojis_to_remove = ['✨', '🚀', '⚖️', '📊', '🔍', '🏆', '⚙️', '📂', '⚠️', '📝', '💡', '🛡️', '🧠', '🔬', '🎯', '💬', '📄', '📥', '✅', '🧭']
for emoji in emojis_to_remove:
    content = content.replace(emoji, '')

# Clean up any weird spacing left by emojis (e.g. "  " to " ")
content = content.replace('  ', ' ').replace(' >', '>')

# 2. Fix CSS and add particles
css_patch = '''
    /* Force text color for visibility */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #1a1a1a !important;
    }
    /* Exempt specific items */
    div.stButton > button, div.stButton > button * {
        color: #ffffff !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: #e56b40 !important;
    }
    .retro-subtitle, .retro-subtitle * {
        color: #e56b40 !important;
    }
    .stAlert, .stAlert * {
        color: inherit !important;
    }
'''

content = content.replace('/* Global Styles */', css_patch + '\n    /* Global Styles */')

# 3. Add Particles Injection Function
particles_js = '''
def inject_particles():
    import streamlit.components.v1 as components
    particles_html = """
    <script>
    if (!parent.document.getElementById('particles-js')) {
        const script = parent.document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js";
        script.onload = () => {
            const div = parent.document.createElement('div');
            div.id = 'particles-js';
            div.style.position = 'fixed';
            div.style.top = '0';
            div.style.left = '0';
            div.style.width = '100vw';
            div.style.height = '100vh';
            div.style.zIndex = '-1';
            parent.document.body.prepend(div);
            parent.particlesJS("particles-js", {
              "particles": {
                "number": {"value": 50},
                "color": {"value": "#e56b40"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.3},
                "size": {"value": 3},
                "line_linked": {"enable": true, "distance": 150, "color": "#e56b40", "opacity": 0.2, "width": 1},
                "move": {"enable": true, "speed": 1}
              },
              "interactivity": {
                "events": {
                  "onhover": {"enable": true, "mode": "grab"},
                  "onclick": {"enable": true, "mode": "push"}
                }
              },
              "retina_detect": true
            });
        };
        parent.document.head.appendChild(script);
    }
    </script>
    """
    components.html(particles_html, height=0, width=0)

inject_particles()
'''

content = content.replace('inject_custom_css()\n', 'inject_custom_css()\n' + particles_js)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied emojis removal, css text color fixes, and particles!")
