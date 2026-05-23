import typer
from pathlib import Path
from .core import predict_from_bytes

app = typer.Typer(help="CLI для медицинской классификации рентгенов")


@app.command()
def predict(input_path: str = typer.Argument(..., help="Путь к изображению")):
    """Сделать предсказание для одного изображения."""
    path = Path(input_path)
    if not path.exists():
        typer.echo(f"❌ Файл не найден: {path}")
        raise typer.Exit(code=1)

    image_bytes = path.read_bytes()
    result = predict_from_bytes(image_bytes)

    typer.echo(f"✅ Предсказание: {result['prediction']}")
    typer.echo(f"📊 Уверенность: {result['confidence']}")
    typer.echo(f"📈 Вероятности:")
    for cls, prob in result['probabilities'].items():
        typer.echo(f"   {cls}: {prob}")


if __name__ == "__main__":
    app()