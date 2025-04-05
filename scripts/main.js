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

    // Array to store available videos from videos.json
    let availableVideos = [
        // Default fallback in case videos.json can't be loaded
        { water: 400, xenon: 20, file: 'default.mp4' }
    ];

    // Update slider values on input
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
        const numWater = parseInt(numWaterSlider.value);
        const numXenon = parseInt(numXenonSlider.value);
        
        // Show loading indicator and hide video
        loadingIndicator.classList.remove('hidden');
        videoElement.classList.add('hidden');
        statusMessage.className = '';
        statusMessage.textContent = '';
        
        // Check if the exact video exists
        checkExactVideo(numWater, numXenon);
    });
    
    // Function to check if the exact video exists
    function checkExactVideo(numWater, numXenon) {
        // First, check if we have this exact video in our available videos list
        const exactVideo = availableVideos.find(video => 
            video.water === numWater && video.xenon === numXenon
        );
        
        if (exactVideo) {
            // We have this exact video, show it
            showVideo(`assets/${exactVideo.file}`, 
                `Showing simulation with ${numWater} water molecules and ${numXenon} xenon atoms.`);
            return;
        }
        
        // If we don't have the exact video, try to fetch it directly to see if it exists
        // but GitHub Actions may have created it since we loaded the page
        const videoUrl = `assets/simulation_w${numWater}_x${numXenon}.mp4`;
        
        fetch(videoUrl, { method: 'HEAD' })
            .then(response => {
                if (response.ok) {
                    // Video exists but wasn't in our list - add it
                    availableVideos.push({
                        water: numWater,
                        xenon: numXenon,
                        file: `simulation_w${numWater}_x${numXenon}.mp4`
                    });
                    
                    // Show the video
                    showVideo(videoUrl, `Showing simulation with ${numWater} water molecules and ${numXenon} xenon atoms.`);
                } else {
                    // Video doesn't exist - find closest match
                    findClosestVideo(numWater, numXenon);
                }
            })
            .catch(error => {
                // Error checking for the video - find closest match
                console.error('Error checking video:', error);
                findClosestVideo(numWater, numXenon);
            });
    }
    
    // Function to find the closest available video
    function findClosestVideo(numWater, numXenon) {
        if (availableVideos.length === 0) {
            // No videos available - show error
            loadingIndicator.classList.add('hidden');
            statusMessage.className = 'error';
            statusMessage.textContent = 'No simulation videos available.';
            return;
        }
        
        // Find closest match using Euclidean distance
        let closestVideo = availableVideos[0];
        let minDistance = Math.sqrt(
            Math.pow(numWater - availableVideos[0].water, 2) + 
            Math.pow(numXenon - availableVideos[0].xenon, 2)
        );
        
        for (let i = 1; i < availableVideos.length; i++) {
            const distance = Math.sqrt(
                Math.pow(numWater - availableVideos[i].water, 2) + 
                Math.pow(numXenon - availableVideos[i].xenon, 2)
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                closestVideo = availableVideos[i];
            }
        }
        
        // Display status message about the exact simulation not being available
        statusMessage.className = 'info';
        statusMessage.textContent = `The exact simulation (${numWater} water, ${numXenon} xenon) hasn't been generated yet. `;
        
        // Show instructions for generating the exact simulation
        statusMessage.textContent += `To generate it, go to the GitHub repository Actions tab and manually trigger the workflow.`;
        
        // Show the closest video
        showVideo(`assets/${closestVideo.file}`, 
            `Showing closest available simulation: ${closestVideo.water} water molecules and ${closestVideo.xenon} xenon atoms.`);
    }
    
    // Function to show a video
    function showVideo(videoUrl, message) {
        // Hide loading indicator and show video
        loadingIndicator.classList.add('hidden');
        videoElement.classList.remove('hidden');
        
        // Add cache-busting parameter to URL
        const timestamp = new Date().getTime();
        videoElement.src = `${videoUrl}?t=${timestamp}`;
        
        // Set status message
        statusMessage.className = 'success';
        statusMessage.textContent = message;
        
        // Load and play the video
        videoElement.load();
        videoElement.play();
    }
    
    // Function to load available videos from videos.json
    function loadAvailableVideos() {
        fetch('assets/videos.json')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load videos.json');
                }
                return response.json();
            })
            .then(data => {
                if (data && data.videos && Array.isArray(data.videos)) {
                    // Update the available videos array
                    availableVideos = data.videos;
                    console.log('Loaded videos:', availableVideos);
                }
            })
            .catch(error => {
                console.error('Error loading videos.json:', error);
                statusMessage.className = 'error';
                statusMessage.textContent = 'Error loading available simulations. Using default video.';
                
                // Make sure we at least have the default video in the list
                if (!availableVideos.some(v => v.file === 'default.mp4')) {
                    availableVideos = [{ water: 400, xenon: 20, file: 'default.mp4' }];
                }
            });
    }
    
    // Load available videos when the page loads
    loadAvailableVideos();
    
    // Also check for default.mp4 explicitly
    fetch('assets/default.mp4', { method: 'HEAD' })
        .then(response => {
            if (!response.ok) {
                throw new Error('Default video not found');
            }
        })
        .catch(error => {
            console.error('Default video not available:', error);
            statusMessage.className = 'error';
            statusMessage.textContent = 'Warning: Default simulation video not found.';
        });
});
