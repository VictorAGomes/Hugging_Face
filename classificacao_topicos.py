from transformers import pipeline

def classificar_topico():
    classificador = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    print("\n=== Classificação de Tópicos ===")
    while True:
        texto = input("\nDigite o texto (em inglês) para classificar (ou 'sair' para encerrar): ")
        if texto.lower() == 'sair':
            print("Encerrando...")
            break

        topicos = ["Technology", "Sports", "Politics", "Entertainment", "Science", "Business"]

        resultado = classificador(texto, topicos)

        print("\nClassificação de Tópicos:")
        for label, score in zip(resultado["labels"], resultado["scores"]):
            print(f"Tópico: {label}, Confiança: {score * 100:.2f}%")

if __name__ == "__main__":
    classificar_topico()
