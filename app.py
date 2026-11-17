"""Public Maple Physio no-show prediction page."""

import gradio as gr

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor

# The assignment's free ZeroGPU setup requires one decorated function.
# The scikit-learn model itself does not need a GPU.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def _gpu_placeholder():
        return None


predictor = NoShowPredictor()  # Loads the model only once, not on every click.


def check_booking(*values):
    data = dict(zip(Appointment.FIELDS, values, strict=True))
    try:
        appointment = Appointment(**data)
        result = predictor.predict(appointment)
    except (ValueError, TypeError) as error:
        return "", "", f"INVALID INPUT: {error}", "", ""

    return (
        f"{result['probability_no_show']:.1%}",
        f"{result['probability_attend']:.1%}",
        result["action"],
        f"{result['threshold']:.2f}",
        "; ".join(result["warnings"]) or "No warnings.",
    )


with gr.Blocks(title="Maple Physio | No-Show Check") as demo:
    gr.Markdown("# Maple Physio | No-Show Check")
    gr.Markdown("Estimate whether to call a patient before an appointment.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Appointment details")
            inputs = []
            for field in Appointment.FIELDS:
                label = field.replace("_", " ").title()
                if field in Appointment.CATEGORIES:
                    component = gr.Dropdown(
                        choices=list(Appointment.CATEGORIES[field]),
                        value=None,
                        label=label,
                    )
                else:
                    low, high = Appointment.NUMERIC_RULES[field]
                    optional = " (optional)" if field == "distance_km" else ""
                    component = gr.Number(
                        label=f"{label} ({low:g} to {high:g}){optional}",
                        value=None,
                    )
                inputs.append(component)

        with gr.Column():
            gr.Markdown("### Prediction results")
            no_show_out = gr.Textbox(label="Chance of no-show", interactive=False)
            attend_out = gr.Textbox(label="Chance of attending", interactive=False)
            action_out = gr.Textbox(label="Recommended action", interactive=False)
            threshold_out = gr.Textbox(label="Decision threshold", interactive=False)
            warning_out = gr.Textbox(label="Warnings and validation", interactive=False)

    button = gr.Button("Check Appointment", variant="primary")
    button.click(
        fn=check_booking,
        inputs=inputs,
        outputs=[no_show_out, attend_out, action_out, threshold_out, warning_out],
    )
    gr.Markdown("Educational demonstration using synthetic appointment data.")


if __name__ == "__main__":
    demo.launch()
