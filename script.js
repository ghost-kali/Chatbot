
const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");

let userMessage = null; // Variable to store user's message
 // Paste your API key here
const inputInitHeight = chatInput.scrollHeight;

const createChatLi = (message, className) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? ` <p></p>` : `<span class=""><img src="./IMG_20240503_090818-removebg-preview.png" height="35vh"  alt=""></span><p></p>`;
    chatLi.innerHTML = chatContent;
    const messageWithLinks = message.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">Click here</a>');
    
    chatLi.querySelector("p").innerHTML = messageWithLinks;
    
    return chatLi; // return chat <li> element
}

const createLoadingLi = (className) =>{

    const LoadingLi = document.createElement("li");
    LoadingLi.classList.add("chat",`${className}`);
    let chatContent = ` <div class="messages__item messages__item--typing" id="GFG_DIV">
    <span class="messages__dot"></span>
    <span class="messages__dot"></span>
    <span class="messages__dot"></span>
</div>`
LoadingLi.innerHTML = chatContent;
return LoadingLi;

}
const genTest = () => {
    const incomingChatLi = createChatLi('', "incoming");
    const messageElement = incomingChatLi.querySelector("p");
    messageElement.textContent = "Hello";
    return incomingChatLi;
}



const generateResponse = (incomingChatLi) => {
   
    const messageElement = incomingChatLi.querySelector("p");

    // Define the properties and message for the API request
 

    // Send POST request to API, get response and set the reponse as paragraph text
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: JSON.stringify({ message: userMessage }),
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json'
        },
      }).then(res => res.json()).then(data => {
        let div = document.getElementById('GFG_DIV');
         div.parentNode.removeChild(div);
        messageElement.innerHTML = data.answer.trim().replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">Click here</a>');;
        chatbox.appendChild(incomingChatLi);
    }).catch(() => {
           let div = document.getElementById('GFG_DIV');
         div.parentNode.removeChild(div);
        messageElement.classList.add("error");
        messageElement.textContent = "Oops! Something went wrong. Please try again.";
        chatbox.appendChild(incomingChatLi);
    }).finally(() => {chatbox.scrollTo(0, chatbox.scrollHeight);});

}

const handleChat = () => {
    userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
    if(!userMessage) return;

    // Clear the input textarea and set its height to default
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    // Append the user's message to the chatbox
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);
    chatbox.appendChild(createLoadingLi("outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);
    const incomingChatLi = createChatLi('', "incoming");
    generateResponse(incomingChatLi);
    setTimeout(() => {
        // Display "Thinking..." message while waiting for the response
        
       
        
        
    }, 10000);
}

chatInput.addEventListener("input", () => {
    // Adjust the height of the input textarea based on its content
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    // If Enter key is pressed without Shift key and the window 
    // width is greater than 800px, handle the chat
    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleChat();
    }
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));