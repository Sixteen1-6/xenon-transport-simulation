document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const numWaterSlider = document.getElementById('numWater');
    const numWaterValue = document.getElementById('numWaterValue');
    const numXenonSlider = document.getElementById('numXenon');
    const numXenonValue = document.getElementById('numXenonValue');
    const simulationForm = document.getElementById('simulation-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    const videoElement = document.getElementById('simulation-video');
    const statusMessage = document.getElementById('status-message');

    // Update slider values
    numWaterSlider.addEventListener('input', function() {
        numWaterValue.textContent = this.value;
    });

    numXenonSlider.addEventListener('input', function() {
        numXenonValue.textContent = this.value;
    });

    // Form submission handler
    simulationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form values
        const numWater = numWaterSlider.value;
        const numXenon = numXenonSlider.value;
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        videoElement.classList.add('hidden');
        statusMessage.className = '';
        statusMessage.textContent = '';
        
        // In a real implementation, you would send these parameters to your backend
        // For GitHub Pages, we'll use a simplified approach that doesn't require a backend
        
        // Simulate a request to generate the video
        simulateVideoGeneration(numWater, numXenon);
    });
    
    // Function to simulate video generation (since we can't run Python directly)
    function simulateVideoGeneration(numWater, numXenon) {
        // In a real implementation, this would be an API call to your backend
        
        // For demonstration, we'll simulate a delay and then show a pre-generated video
        setTimeout(function() {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
            videoElement.classList.remove('hidden');
            
            // Update the video source to a pre-generated video based on parameters
            // In a real implementation, this would be a dynamically generated video
            const timestamp = new Date().getTime(); // Cache-busting
            
            // Here we're just using the default video, but in reality
            // you would point to different videos based on the parameters
            videoElement.src = `assets/default.mp4?t=${timestamp}`;
            
            // Display success message
            statusMessage.className = 'success';
            statusMessage.textContent = `Simulation generated with ${numWater} water molecules and ${numXenon} xenon atoms.`;
            
            // Reload the video to show the "new" content
            videoElement.load();
            videoElement.play();
        }, 3000); // Simulate 3 second delay
    }
});
