from transformers import pipeline

def gerar_texto():
    gerador = pipeline("text-generation", model="gpt2", truncation=True)

    print("\n=== Geração de Texto ===")
    while True:
        prompt = input("\nDigite um texto inicial (em inglês) (ou 'sair' para encerrar): ")
        if prompt.lower() == 'sair':
            print("Encerrando...")
            break

        resultado = gerador(prompt, max_length=50, num_return_sequences=1, pad_token_id=50256)

        texto_gerado = resultado[0]['generated_text']

        print(f"\nTexto Gerado:\n{texto_gerado}")

if __name__ == "__main__":
    gerar_texto()
