const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { GithubRepoLoader } = require("langchain/document_loaders/web/github");
const { SystemChatMessage, HumanChatMessage } = require("langchain/schema");
require("dotenv").config();

const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function getUserQuestion() {
  return new Promise((resolve) => {
    rl.question("Ask a question: ", (answer) => {
      resolve(answer);
    });
  });
}

const loader = new GithubRepoLoader(
  "https://github.com/ashgaliyev/chatgpt-telegram",
  { branch: "main", recursive: false, unknown: "warn" }
);

async function getContext(vectorStore, question) {
  const results = await vectorStore.similaritySearch(question, 1);
  return results.map(res => {
    return `**FILE: ${res.metadata.source}**\n${res.pageContent}\n\n`
  })
}

async function main() {
  const docs = await loader.load();
  const vectorStor = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  const chat = new ChatOpenAI();
  const history = [
    new SystemChatMessage(
      "You're helpful assistant which can answer any question about the code."
    ),
  ];

  async function loop() {
    const userQuestion = await getUserQuestion();
    const context = await getContext(vectorStor, userQuestion);

    const preparedContext = `
    Here is the code context for the question:\n
    ${context.join("\n")}
    `;
    const humanMessage = new HumanChatMessage(`${userQuestion}\n${preparedContext}`);
    const response = await chat.call([...history, humanMessage]);
    history.push(new HumanChatMessage(userQuestion));

    console.log(response);
    loop();
  }
  loop();
};

main();
