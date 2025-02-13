from transformers import pipeline

def analisar_sentimentos():
    analisador = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    print("\n=== Análise de Sentimentos ===")
    while True:
        texto = input("\nDigite um texto(em inglês) para análise (ou 'sair' para encerrar): ")
        if texto.lower() == 'sair':
            print("Encerrando...")
            break

        resultado = analisador(texto)

        sentimento = resultado[0]['label']
        confianca = resultado[0]['score']

        print(f"\nSentimento: {sentimento}")
        print(f"Confiança: {confianca * 100:.2f}%")

if __name__ == "__main__":
    analisar_sentimentos()