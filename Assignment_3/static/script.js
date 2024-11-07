# static/script.js
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

function loadText() {
    const inputText = document.getElementById("inputText").value;
    document.getElementById("loadedText").value = inputText;
}

function preprocessText() {
    const text = document.getElementById("loadedText").value;
    const operation = document.getElementById("preprocessOperation").value;
    let result = text;
    
    switch(operation) {
        case "tokenize":
            result = text.split(/\s+/).join('\n');
            break;
        case "lowercase":
            result = text.toLowerCase();
            break;
        case "uppercase":
            result = text.toUpperCase();
            break;
        case "removeIndent":
            result = text.replace(/^\s+/gm, '');
            break;
    }
    
    document.getElementById("processedText").value = result;
}

function augmentText() {
    const text = document.getElementById("processedText").value;
    const operation = document.getElementById("augmentOperation").value;
    // In a real application, you would implement more sophisticated augmentation
    // This is just a simple example
    let result = text;
    
    if (operation === "swap") {
        const words = text.split(' ');
        for (let i = 0; i < words.length - 1; i += 2) {
            [words[i], words[i + 1]] = [words[i + 1], words[i]];
        }
        result = words.join(' ');
    }
    
    document.getElementById("augmentedText").value = result;
}