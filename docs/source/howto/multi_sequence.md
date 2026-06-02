# Modo multi-secuencia

Triage-MRI soporta cargar múltiples secuencias como canales de entrada.

## Activar

En `configs/data.yaml`:

```yaml
data:
  multi_sequence:
    enabled: true
    input_sequences: ["t1", "t1ce", "t2", "flair"]
```

La primera capa `patch_embed.proj` del encoder se adapta automáticamente
de 1 a N canales (Kaiming init + peso Triad en canal 0).

## Entrenar

```bash
triage-train --multi_sequence --output_dir outputs/run_ms
```

Los volúmenes cargados tienen shape `(4, 96, 96, 96)`.

## Limitaciones

- Requiere que cada caso tenga todas las secuencias configuradas.
- Si falta alguna, se usan solo las disponibles.
- El encoder congelado espera features normalizadas a [0, 1] por secuencia.
