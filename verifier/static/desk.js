const form = document.getElementById("packet");
const result = document.getElementById("result");
const kicker = document.getElementById("kicker");
const stampWord = document.getElementById("stamp-word");
const story = document.getElementById("story");
const facts = document.getElementById("facts");
const factTier = document.getElementById("fact-tier");
const factP = document.getElementById("fact-p");
const factC = document.getElementById("fact-c");
const button = document.getElementById("stamp");

function pct(value) {
  if (value === null || value === undefined) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  kicker.textContent = "Checking the seal";
  stampWord.textContent = "…";
  story.textContent = "Talking to EZKL. Still not opening the loan.";
  result.dataset.state = "idle";
  facts.hidden = true;

  try {
    const response = await fetch("/verify", { method: "POST", body: new FormData(form) });
    const data = await response.json();
    result.dataset.state = data.ok ? "pass" : "fail";
    kicker.textContent = data.ok ? "Seal holds" : "Seal broken";
    stampWord.textContent = data.ok ? "VERIFIED" : "REJECTED";
    story.textContent = data.reason;
    facts.hidden = !data.ok;
    factTier.textContent = data.risk_tier || "—";
    factP.textContent = pct(data.default_probability);
    factC.textContent = data.commitment || "—";
  } catch (err) {
    result.dataset.state = "fail";
    kicker.textContent = "Desk error";
    stampWord.textContent = "REJECTED";
    story.textContent = String(err);
  } finally {
    button.disabled = false;
  }
});
