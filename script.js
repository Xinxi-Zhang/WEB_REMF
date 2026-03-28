document.addEventListener("DOMContentLoaded", function () {
    // Function to handle smooth scrolling
    function scrollToSection(event) {
        event.preventDefault(); // Prevent default anchor behavior
        
        // Get the target section's ID from the clicked link's href attribute
        const targetId = this.getAttribute("href").substring(1);
        const targetSection = document.getElementById(targetId);

        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    }

    // Attach click event listeners to navbar items
    document.querySelector('a[href="#people"]').addEventListener("click", scrollToSection);
    document.querySelector('a[href="#research"]').addEventListener("click", scrollToSection);
});

document.addEventListener("DOMContentLoaded", function () {
    // Get all the clickable text elements
    const toggleTexts = document.querySelectorAll(".toggle-abstract");

    toggleTexts.forEach(toggle => {
        toggle.addEventListener("click", function () {
            const pubContainer = this.closest(".pub-container");
            const shortAbstract = pubContainer.querySelector(".short-abstract");
            const fullAbstract = pubContainer.querySelector(".full-abstract");
            const showFullText = pubContainer.querySelector(".show-full");
            const showTldrText = pubContainer.querySelector(".show-tldr");

            // Toggle visibility
            if (shortAbstract.style.display === "none") {
                // Switch back to TL;DR mode
                shortAbstract.style.display = "block";
                fullAbstract.style.display = "none";
                showFullText.style.display = "inline"; // Show "Show Full Abstract"
                showTldrText.style.display = "none"; // Hide "Show TL;DR"
            } else {
                // Switch to full abstract mode
                shortAbstract.style.display = "none";
                fullAbstract.style.display = "block";
                showFullText.style.display = "none"; // Hide "Show Full Abstract"
                showTldrText.style.display = "inline"; // Show "Show TL;DR"
            }
        });
    });
});