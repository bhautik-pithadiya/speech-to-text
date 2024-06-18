document.addEventListener('DOMContentLoaded', async () => {
    const idList = document.getElementById('id-list');
    const chatBox = document.querySelector('.form_data');
    const audioForm = document.querySelector('.audioform');
    const resultContainer = document.querySelector('.result_container');
    const resultParagraph = document.getElementById('result');

    let jsonData = [];

    // Fetch JSON data
    try {
        const response = await fetch('/data/results/result.json');
        jsonData = await response.json();

        // Sort jsonData based on the DateTime property in descending order
        jsonData.sort((a, b) => new Date(b.DateTime) - new Date(a.DateTime));

        // Populate ID list
        jsonData.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item.id;
            li.addEventListener('click', () => {
                displayUserData(item);
            });
            idList.appendChild(li);
        });
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }

    // Function to display user data
    function displayUserData(userItem) {
        audioForm.style.display = 'none';
        resultContainer.style.display = 'block';
        const audioFilePath = `/data/audios/${userItem.id}.wav`;
        resultParagraph.innerHTML = `
        <div class='audio-center'>    
        <audio controls>
                <source src="${audioFilePath}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            </div>
            <p><strong>Transcript:</strong> ${userItem.Transcript}</p>
            <p><strong>Summary:</strong> ${userItem.Summary}</p>
            <p><strong>Sentiment:</strong> ${userItem.Sentiment}</p>
        `;
    }

    // New chat button functionality
    document.getElementById('new-chat').addEventListener('click', function() {
        chatBox.style.display = 'block';
        audioForm.style.display = 'block';
        resultContainer.style.display = 'none';
        resultParagraph.innerHTML = '';
        document.getElementById('audioUploadForm').reset();
    });

    // Audio submission form handling
    document.getElementById('audioUploadForm').addEventListener('submit', async function(event) {
        event.preventDefault();
        const form = event.target;
        const formData = new FormData(form);
        const audioFile = formData.get('audioFile');

        // Create a FormData object to send the audio file to the server
        const formDataToSend = new FormData();
        formDataToSend.append('audioFile', audioFile);

        try {
            const response = await fetch(form.action, {
                method: 'POST',
                body: formDataToSend,
            });

            if (response.ok) {
                const result = await response.json();
                // Display the generated text in the result container
                document.getElementById('result').textContent = result.text;
            } else {
                // Handle error responses
                document.getElementById('result').textContent = `Error: ${response.statusText}`;
            }
        } catch (error) {
            // Handle network errors
            document.getElementById('result').textContent = `Error: ${error.message}`;
        }
    });
});
