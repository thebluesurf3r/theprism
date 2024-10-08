document.addEventListener("DOMContentLoaded", function() {
    // Smooth scrolling for links (if any)
    const smoothScrollLinks = document.querySelectorAll('a[href^="#"]');
    smoothScrollLinks.forEach(link => {
        link.addEventListener("click", function(event) {
            event.preventDefault();
            const targetID = this.getAttribute("href");
            const targetElement = document.querySelector(targetID);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop,
                    behavior: "smooth"
                });
            }
        });
    });

    // Add interactive hover effects with JS for smoother transitions
    const interactiveElements = document.querySelectorAll('.button, ul li');
    interactiveElements.forEach(element => {
        element.addEventListener("mouseover", function() {
            this.style.transition = "transform 0.3s ease, background-color 0.3s ease";
            this.style.transform = "scale(1.05)";
        });

        element.addEventListener("mouseout", function() {
            this.style.transform = "scale(1)";
        });

        // Clickable buttons with hover effect feedback
        element.addEventListener("click", function() {
            this.style.transition = "transform 0.1s ease";
            this.style.transform = "scale(0.95)";
            setTimeout(() => {
                this.style.transform = "scale(1)";
            }, 100);
        });
    });

    // Floating hover effect for entire sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.addEventListener("mouseover", function() {
            this.style.boxShadow = "0 8px 16px rgba(0, 0, 0, 0.3)";
            this.style.transition = "box-shadow 0.3s ease";
        });

        section.addEventListener("mouseout", function() {
            this.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.1)";
        });
    });

    // Simulating Bootstrap Vue-like interactive components
    const collapsibleSections = document.querySelectorAll(".collapsible");
    collapsibleSections.forEach(collapsible => {
        collapsible.addEventListener("click", function() {
            this.classList.toggle("active");

            const content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null; // Close
            } else {
                content.style.maxHeight = content.scrollHeight + "px"; // Open
            }
        });
    });
});
