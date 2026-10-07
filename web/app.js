// Chat page for POST /api/ask. No framework and no dependencies: one form, one
// streamed answer at a time.
//
// The API streams Server-Sent Events over a POST, which the browser's
// EventSource cannot send, so the stream is read with fetch and parsed here.
// Event contract: see the module docstring in src/rag_me/api.py.
//
// All answer and source text is inserted with textContent, never innerHTML, so
// nothing the model or a visitor writes can become markup on this page.

"use strict";

const MAX_QUESTION_LENGTH = 500;

const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const conversation = document.getElementById("conversation");
const suggestions = document.getElementById("suggestions");
const exchangeTemplate = document.getElementById("exchange-template");
const hint = document.getElementById("hint");
const defaultHintText = hint.textContent;

let isAnswering = false;

// --- events ---------------------------------------------------------------------

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  resizeQuestionInput();
  clearInputError();
});

suggestions.addEventListener("click", (event) => {
  const suggestion = event.target.closest(".suggestion");
  if (suggestion) submitQuestion(suggestion.textContent);
});

// --- flow -----------------------------------------------------------------------

async function submitQuestion(rawQuestion) {
  const question = rawQuestion.trim();
  if (isAnswering || question === "") return;
  if (question.length > MAX_QUESTION_LENGTH) {
    // Shown at the input, not as an exchange: the question stays editable there.
    showInputError(`Please keep questions under ${MAX_QUESTION_LENGTH} characters.`);
    return;
  }

  setAnswering(true);
  suggestions.hidden = true;
  questionInput.value = "";
  resizeQuestionInput();
  const exchange = createExchange(question);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      showExchangeError(exchange, await describeErrorResponse(response));
      return;
    }
    await readAnswerStream(response, exchange);
  } catch {
    showExchangeError(exchange, "Couldn't reach the server. Check your connection and try again.");
  } finally {
    finishExchange(exchange);
    setAnswering(false);
    questionInput.focus();
  }
}

async function readAnswerStream(response, exchange) {
  let sources = [];
  let receivedTerminalEvent = false;

  for await (const { event, data } of readServerSentEvents(response.body)) {
    if (event === "sources") {
      sources = data.sources;
    } else if (event === "delta") {
      exchange.thinking.hidden = true;
      exchange.answerText.textContent += data.text;
    } else if (event === "done") {
      showCitedSources(exchange, sources, data.cited);
      receivedTerminalEvent = true;
    } else if (event === "error") {
      showExchangeError(exchange, data.detail);
      receivedTerminalEvent = true;
    }
  }

  // A stream that ends without done/error was cut off (network drop, timeout).
  if (!receivedTerminalEvent) {
    showExchangeError(exchange, "The answer was cut off. Please try again.");
  }
}

// Yields {event, data} objects from a text/event-stream body. Handles events
// split across network chunks and both LF and CRLF line endings.
async function* readServerSentEvents(body) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replaceAll("\r\n", "\n");
    let separatorIndex;
    while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      const parsed = parseServerSentEvent(block);
      if (parsed) yield parsed;
    }
  }
}

function parseServerSentEvent(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

async function describeErrorResponse(response) {
  // Errors are RFC 9457 Problem Details; their `detail` is written for visitors.
  try {
    const problem = await response.json();
    if (typeof problem.detail === "string" && problem.detail) return problem.detail;
  } catch {
    // Not JSON (e.g. a platform error page); fall through to a generic message.
  }
  return response.status === 429
    ? "Too many questions right now. Please try again in a few minutes."
    : "Something went wrong. Please try again.";
}

// --- rendering ------------------------------------------------------------------

function createExchange(question) {
  const fragment = exchangeTemplate.content.cloneNode(true);
  const element = fragment.querySelector(".exchange");
  const exchange = {
    element,
    answer: element.querySelector(".answer"),
    answerText: element.querySelector(".answer-text"),
    thinking: element.querySelector(".thinking"),
    sources: element.querySelector(".sources"),
  };
  element.querySelector(".question-bubble").textContent = question;
  conversation.append(element);
  element.scrollIntoView({ block: "end", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  return exchange;
}

function showCitedSources(exchange, sources, citedNumbers) {
  const cited = citedNumbers
    .map((number) => sources.find((source) => source.number === number))
    .filter(Boolean);
  if (cited.length === 0) return;

  exchange.sources.querySelector("summary").textContent =
    cited.length === 1 ? "1 source" : `${cited.length} sources`;
  const list = exchange.sources.querySelector(".source-list");
  for (const source of cited) {
    const item = document.createElement("li");
    item.value = source.number; // keeps the list number equal to the [n] in the text
    item.textContent = source.title;
    list.append(item);
  }
  exchange.sources.hidden = false;
}

function showExchangeError(exchange, message) {
  exchange.answer.classList.add("is-error");
  exchange.answerText.textContent = message;
  exchange.sources.hidden = true;
}

function finishExchange(exchange) {
  exchange.thinking.hidden = true;
  exchange.answer.setAttribute("aria-busy", "false");
}

function showInputError(message) {
  hint.textContent = message;
  hint.classList.add("is-error");
  questionInput.setAttribute("aria-invalid", "true");
}

function clearInputError() {
  if (!hint.classList.contains("is-error")) return;
  hint.textContent = defaultHintText;
  hint.classList.remove("is-error");
  questionInput.removeAttribute("aria-invalid");
}

function setAnswering(answering) {
  isAnswering = answering;
  askButton.disabled = answering;
  askButton.textContent = answering ? "Answering…" : "Ask";
}

function resizeQuestionInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${questionInput.scrollHeight}px`;
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
