import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def explain_results(training_results, dataset_info, target_col):
    try:
        best = training_results.get("best_model", "Unknown")
        score = training_results.get("best_score", 0)
        task = training_results.get("task_type", "classification")
        results = training_results.get("results", [])
        features = training_results.get("feature_importance", [])

        comparison = ""
        for r in results[:7]:
            if "error" not in r:
                if task == "classification":
                    comparison += f"- {r['name']}: accuracy={r.get('accuracy', 'N/A')}%, cv_score={r.get('cv_score', 'N/A')}%\n"
                else:
                    comparison += f"- {r['name']}: r2={r.get('r2_score', 'N/A')}%, rmse={r.get('rmse', 'N/A')}\n"

        feat_text = ""
        if features:
            for f in features[:10]:
                feat_text += f"- {f['feature']}: {f['importance']}\n"

        prompt = f"""You are explaining ML results to someone who uploaded a dataset.

Dataset info:
- Rows: {dataset_info.get('total_rows', 'unknown')}
- Features: {dataset_info.get('total_columns', 'unknown')}
- Target column: {target_col}
- Task type: {task}

Model comparison results:
{comparison}

Best model: {best} with {'accuracy' if task == 'classification' else 'R2 score'} of {score}%

Top features by importance:
{feat_text if feat_text else 'Not available'}

Write a clear, helpful explanation that covers:
1. What the best model is and how well it performs (is {score}% good or bad for this type of problem?)
2. Which features matter most and why that makes sense
3. How the models compare to each other
4. Practical advice - what can the user do with these results
5. Any warnings or things to watch out for

Keep it conversational and easy to understand. Use bullet points where helpful.
Dont use markdown headers, just plain text with bullet points."""

        chat = get_client().chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a friendly data scientist explaining ML results to a non-technical person. Be clear, practical, and honest about limitations."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL,
            temperature=0.7,
            max_tokens=1024
        )

        return {
            "explanation": chat.choices[0].message.content,
            "best_model": best,
            "score": score
        }

    except Exception as e:
        print(f"explainer error: {e}")
        return {
            "explanation": f"Could not generate AI explanation: {str(e)}",
            "best_model": training_results.get("best_model", "Unknown"),
            "score": training_results.get("best_score", 0)
        }
