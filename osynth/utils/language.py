"""Language-specific configuration for multilingual data generation.

Provides ``LANG_CONFIG``, a mapping from language code to a dict of
language-specific content used in Jinja2 prompt templates. Each entry
contains filler words, backchannels, accent examples, email conventions,
spoken-form rules, and other locale-sensitive content.

Usage in prompts::

    {{ lang.filler_words | join('", "') }}
    {{ lang.language_instruction }}
"""
from __future__ import annotations

LANG_CONFIG: dict[str, dict] = {
    "en": {
        "code": "en",
        "name": "English",
        "language_instruction": "",  # no extra instruction needed for English
        "content_language_instruction": "",
        # Company / location
        "location_format": "City, State (e.g., San Francisco, CA)",
        "default_location_examples": "San Francisco, CA; Austin, TX; New York, NY; Boston, MA; Seattle, WA",
        # Accents
        "accent_examples": [
            "slight Southern accent",
            "New York accent",
            "Midwestern accent",
            "California casual",
            "neutral American",
        ],
        # Disfluency
        "filler_words": ["um", "uh", "like", "you know", "I mean", "sort of", "kind of", "right", "so", "well", "basically", "actually", "honestly"],
        "hedge_words": ["maybe", "perhaps", "I think", "probably", "kind of", "sort of", "basically", "arguably", "it seems like"],
        "backchannel_phrases": ["mm-hmm", "mhm", "uh-huh", "yeah", "yes", "right", "okay", "sure", "got it", "I see", "totally", "absolutely", "exactly", "makes sense", "interesting", "wow", "for sure", "yep", "oh wow", "no way"],
        # Speaking rate
        "words_per_minute_range": "120-150",
        # Email conventions
        "email_greeting_formal": "Dear [Name],",
        "email_greeting_informal": "Hi [Name],",
        "email_signoff_formal": "Best regards,",
        "email_signoff_informal": "Thanks,",
        "email_conventions_note": "",
        # Working days
        "working_days_label": "Monday through Friday",
        # Spoken form
        "spoken_form_rules": """1. **Dollar amounts**: "$4.11" → "four dollars and eleven cents"
2. **Percentages**: "34%" → "thirty-four percent", "40%" → "forty percent"
3. **Numbers in context**: spell out naturally. "200 to 800" → "two hundred to eight hundred". "7 AM" → "seven A.M.". Keep numbers that are clearly spoken as digits (e.g. phone numbers) in a speakable form.
4. **Acronyms / initialisms**: only expand acronyms that TTS is likely to mispronounce. Well-known acronyms (CEO, API, etc.) can be left as-is if TTS handles them. For uncommon ones, insert periods between letters so TTS spells them out.
5. **Backchannels**: "Mm-hmm." → "mhmm", "Mm-hmm" → "mhmm\"""",
        "spoken_form_filler_examples": """- Example: "We need to finalize the budget" → "We need to, um, finalize the budget"
   - Example: "That's a great point" → "That's, uh, that's a great point\"""",
        "spoken_form_repetition_examples": """- Example: "I think we should" → "I, I think we should"
   - Example: "the timeline is" → "the, the timeline is\"""",
        "spoken_form_false_start_examples": """- Example: "We could launch in March" → "We could do it in, well, we could launch in March"
   - Example: "The blue design" → "The blu-, the red design\"""",
        "spoken_form_prolongation_examples": """- Example: "So that is fine" → "Sooo that is fine"
   - Example: "I mean" → "I meean\"""",
    },
    "de": {
        "code": "de",
        "name": "German",
        "language_instruction": "IMPORTANT: Generate ALL text content in German. Use culturally appropriate German names, company names, and conventions. The workplace is set in a German-speaking country.",
        "content_language_instruction": "Generate all text content (dialogue, descriptions, emails, documents) in German. JSON keys and structural field names remain in English.",
        # Company / location
        "location_format": "City, Germany (e.g., München, Germany)",
        "default_location_examples": "München; Berlin; Hamburg; Frankfurt am Main; Stuttgart; Köln",
        # Accents
        "accent_examples": [
            "leichter bayerischer Akzent",
            "Berliner Dialekt",
            "norddeutscher Akzent",
            "österreichisches Deutsch",
            "Schweizer Hochdeutsch",
        ],
        # Disfluency
        "filler_words": ["äh", "ähm", "halt", "also", "sozusagen", "quasi", "irgendwie", "na ja", "sagen wir mal", "genau", "eben", "ja"],
        "hedge_words": ["vielleicht", "ich denke", "wahrscheinlich", "eigentlich", "im Grunde", "möglicherweise", "gewissermaßen", "ich würde sagen"],
        "backchannel_phrases": ["mhm", "ja", "genau", "stimmt", "ach so", "klar", "okay", "richtig", "verstehe", "ja genau", "absolut", "tatsächlich", "interessant", "na klar", "aha", "ja sicher"],
        # Speaking rate
        "words_per_minute_range": "120-150",
        # Email conventions
        "email_greeting_formal": "Sehr geehrte/r [Name],",
        "email_greeting_informal": "Hallo [Name],",
        "email_signoff_formal": "Mit freundlichen Grüßen,",
        "email_signoff_informal": "Viele Grüße,",
        "email_conventions_note": "Use German email conventions: formal greetings (Sehr geehrte/r) for external contacts, informal (Hallo, Hi) for close colleagues. Sign-offs: Mit freundlichen Grüßen (formal), Viele Grüße / Beste Grüße (informal).",
        # Working days
        "working_days_label": "Montag bis Freitag",
        # Spoken form
        "spoken_form_rules": """1. **Currency**: "4,11€" → "vier Euro und elf Cent", "$50" → "fünfzig Dollar"
2. **Percentages**: "34%" → "vierunddreißig Prozent", "40%" → "vierzig Prozent"
3. **Numbers in context**: spell out naturally in German. "200 bis 800" → "zweihundert bis achthundert". "7 Uhr" → "sieben Uhr".
4. **Acronyms / initialisms**: leave well-known ones (CEO, API, IT) as-is. For uncommon ones, insert periods so TTS spells them out.
5. **Backchannels**: "Mhm." → "mhm", "Mm-hmm" → "mhm\"""",
        "spoken_form_filler_examples": """- Example: "Wir müssen das Budget finalisieren" → "Wir müssen, äh, das Budget finalisieren"
   - Example: "Das ist ein guter Punkt" → "Das ist, ähm, das ist ein guter Punkt\"""",
        "spoken_form_repetition_examples": """- Example: "Ich denke, wir sollten" → "Ich, ich denke, wir sollten"
   - Example: "der Zeitplan ist" → "der, der Zeitplan ist\"""",
        "spoken_form_false_start_examples": """- Example: "Wir könnten im März launchen" → "Wir könnten das im, also, wir könnten im März launchen"
   - Example: "Das blaue Design" → "Das blau-, das rote Design\"""",
        "spoken_form_prolongation_examples": """- Example: "Also das ist gut" → "Alsooo das ist gut"
   - Example: "Ich meine" → "Ich meeine\"""",
    },
    "es": {
        "code": "es",
        "name": "Spanish",
        "language_instruction": "IMPORTANT: Generate ALL text content in Spanish. Use culturally appropriate Spanish names, company names, and conventions. The workplace is set in a Spanish-speaking country.",
        "content_language_instruction": "Generate all text content (dialogue, descriptions, emails, documents) in Spanish. JSON keys and structural field names remain in English.",
        # Company / location
        "location_format": "City, Spain (e.g., Madrid, Spain)",
        "default_location_examples": "Madrid; Barcelona; Valencia; Sevilla; Bilbao; Málaga",
        # Accents
        "accent_examples": [
            "acento madrileño",
            "acento catalán",
            "acento andaluz",
            "acento mexicano",
            "acento argentino",
        ],
        # Disfluency
        "filler_words": ["eh", "este", "o sea", "bueno", "pues", "digamos", "como que", "a ver", "es que", "mira", "entonces", "vale"],
        "hedge_words": ["quizás", "tal vez", "creo que", "probablemente", "básicamente", "en cierto modo", "diría que", "posiblemente"],
        "backchannel_phrases": ["ajá", "sí", "claro", "vale", "exacto", "ya", "ok", "sí sí", "por supuesto", "efectivamente", "entiendo", "de acuerdo", "correcto", "interesante", "ah vale"],
        # Speaking rate
        "words_per_minute_range": "150-180",
        # Email conventions
        "email_greeting_formal": "Estimado/a [Name],",
        "email_greeting_informal": "Hola [Name],",
        "email_signoff_formal": "Atentamente,",
        "email_signoff_informal": "Un saludo,",
        "email_conventions_note": "Use Spanish email conventions: formal greetings (Estimado/a) for external contacts, informal (Hola, Buenos días) for colleagues. Sign-offs: Atentamente / Cordialmente (formal), Un saludo / Saludos (informal).",
        # Working days
        "working_days_label": "lunes a viernes",
        # Spoken form
        "spoken_form_rules": """1. **Currency**: "4,11€" → "cuatro euros con once céntimos", "$50" → "cincuenta dólares"
2. **Percentages**: "34%" → "treinta y cuatro por ciento", "40%" → "cuarenta por ciento"
3. **Numbers in context**: spell out naturally in Spanish. "200 a 800" → "doscientos a ochocientos". "7 AM" → "siete de la mañana".
4. **Acronyms / initialisms**: leave well-known ones (CEO, API, IT) as-is. For uncommon ones, insert periods so TTS spells them out.
5. **Backchannels**: "Ajá." → "ajá", "Mm-hmm" → "mhm\"""",
        "spoken_form_filler_examples": """- Example: "Necesitamos finalizar el presupuesto" → "Necesitamos, eh, finalizar el presupuesto"
   - Example: "Es un buen punto" → "Es, este, es un buen punto\"""",
        "spoken_form_repetition_examples": """- Example: "Yo creo que deberíamos" → "Yo, yo creo que deberíamos"
   - Example: "el plazo es" → "el, el plazo es\"""",
        "spoken_form_false_start_examples": """- Example: "Podríamos lanzar en marzo" → "Podríamos hacer, bueno, podríamos lanzar en marzo"
   - Example: "El diseño azul" → "El azu-, el diseño rojo\"""",
        "spoken_form_prolongation_examples": """- Example: "Bueno eso está bien" → "Bueeeno eso está bien"
   - Example: "Es que" → "Es quee\"""",
    },
    "fr": {
        "code": "fr",
        "name": "French",
        "language_instruction": "IMPORTANT: Generate ALL text content in French. Use culturally appropriate French names, company names, and conventions. The workplace is set in a French-speaking country.",
        "content_language_instruction": "Generate all text content (dialogue, descriptions, emails, documents) in French. JSON keys and structural field names remain in English.",
        # Company / location
        "location_format": "City, France (e.g., Paris, France)",
        "default_location_examples": "Paris; Lyon; Marseille; Toulouse; Nantes; Bordeaux",
        # Accents
        "accent_examples": [
            "accent parisien",
            "accent du sud de la France",
            "accent lyonnais",
            "accent québécois",
            "accent belge",
        ],
        # Disfluency
        "filler_words": ["euh", "hein", "ben", "genre", "enfin", "tu vois", "en fait", "du coup", "voilà", "bah", "quoi", "bon"],
        "hedge_words": ["peut-être", "je pense", "probablement", "en gros", "disons", "je dirais", "il me semble", "éventuellement"],
        "backchannel_phrases": ["mhm", "oui", "d'accord", "voilà", "exactement", "ok", "ah oui", "c'est ça", "bien sûr", "tout à fait", "je vois", "effectivement", "absolument", "intéressant", "ah bon"],
        # Speaking rate
        "words_per_minute_range": "140-170",
        # Email conventions
        "email_greeting_formal": "Cher/Chère [Name],",
        "email_greeting_informal": "Bonjour [Name],",
        "email_signoff_formal": "Cordialement,",
        "email_signoff_informal": "Bien à vous,",
        "email_conventions_note": "Use French email conventions: formal greetings (Cher/Chère, Madame/Monsieur) for external contacts, informal (Bonjour, Salut) for close colleagues. Sign-offs: Cordialement / Bien cordialement (formal), Bien à vous / À bientôt (informal).",
        # Working days
        "working_days_label": "du lundi au vendredi",
        # Spoken form
        "spoken_form_rules": """1. **Currency**: "4,11€" → "quatre euros et onze centimes", "$50" → "cinquante dollars"
2. **Percentages**: "34%" → "trente-quatre pour cent", "40%" → "quarante pour cent"
3. **Numbers in context**: spell out naturally in French. "200 à 800" → "deux cents à huit cents". "7h" → "sept heures".
4. **Acronyms / initialisms**: leave well-known ones (PDG, DRH, API) as-is. For uncommon ones, insert periods so TTS spells them out.
5. **Backchannels**: "Mhm." → "mhm", "Mm-hmm" → "mhm\"""",
        "spoken_form_filler_examples": """- Example: "Nous devons finaliser le budget" → "Nous devons, euh, finaliser le budget"
   - Example: "C'est un bon point" → "C'est, euh, c'est un bon point\"""",
        "spoken_form_repetition_examples": """- Example: "Je pense qu'on devrait" → "Je, je pense qu'on devrait"
   - Example: "le planning est" → "le, le planning est\"""",
        "spoken_form_false_start_examples": """- Example: "On pourrait lancer en mars" → "On pourrait faire, enfin, on pourrait lancer en mars"
   - Example: "Le design bleu" → "Le ble-, le design rouge\"""",
        "spoken_form_prolongation_examples": """- Example: "Donc c'est bien" → "Doonc c'est bien"
   - Example: "Je veux dire" → "Je veux diire\"""",
    },
    "pt": {
        "code": "pt",
        "name": "Portuguese",
        "language_instruction": "IMPORTANT: Generate ALL text content in Brazilian Portuguese. Use culturally appropriate Brazilian names, company names, and conventions. The workplace is set in Brazil.",
        "content_language_instruction": "Generate all text content (dialogue, descriptions, emails, documents) in Brazilian Portuguese. JSON keys and structural field names remain in English.",
        # Company / location
        "location_format": "City, State (e.g., São Paulo, SP)",
        "default_location_examples": "São Paulo, SP; Rio de Janeiro, RJ; Belo Horizonte, MG; Curitiba, PR; Porto Alegre, RS; Brasília, DF",
        # Accents
        "accent_examples": [
            "sotaque paulistano",
            "sotaque carioca",
            "sotaque mineiro",
            "sotaque nordestino",
            "sotaque sulista",
        ],
        # Disfluency
        "filler_words": ["é", "ah", "tipo", "né", "então", "assim", "tipo assim", "sabe", "bom", "olha", "veja", "pois é"],
        "hedge_words": ["talvez", "acho que", "provavelmente", "basicamente", "de certa forma", "eu diria que", "possivelmente", "digamos"],
        "backchannel_phrases": ["mhm", "sim", "é", "certo", "tá", "uhum", "ah é", "isso", "exato", "com certeza", "entendi", "claro", "verdade", "interessante", "nossa"],
        # Speaking rate
        "words_per_minute_range": "140-170",
        # Email conventions
        "email_greeting_formal": "Prezado(a) [Name],",
        "email_greeting_informal": "Olá [Name],",
        "email_signoff_formal": "Atenciosamente,",
        "email_signoff_informal": "Abraços,",
        "email_conventions_note": "Use Brazilian Portuguese email conventions: formal greetings (Prezado/a, Caro/a) for external contacts, informal (Olá, Oi) for close colleagues. Sign-offs: Atenciosamente / Cordialmente (formal), Abraços / Até mais (informal).",
        # Working days
        "working_days_label": "segunda a sexta-feira",
        # Spoken form
        "spoken_form_rules": """1. **Currency**: "R$4,11" → "quatro reais e onze centavos", "$50" → "cinquenta dólares"
2. **Percentages**: "34%" → "trinta e quatro por cento", "40%" → "quarenta por cento"
3. **Numbers in context**: spell out naturally in Portuguese. "200 a 800" → "duzentos a oitocentos". "7h" → "sete horas".
4. **Acronyms / initialisms**: leave well-known ones (CEO, TI, RH) as-is. For uncommon ones, insert periods so TTS spells them out.
5. **Backchannels**: "Mhm." → "mhm", "Uhum" → "uhum\"""",
        "spoken_form_filler_examples": """- Example: "Precisamos finalizar o orçamento" → "Precisamos, é, finalizar o orçamento"
   - Example: "Esse é um bom ponto" → "Esse é, ah, esse é um bom ponto\"""",
        "spoken_form_repetition_examples": """- Example: "Eu acho que devemos" → "Eu, eu acho que devemos"
   - Example: "o prazo é" → "o, o prazo é\"""",
        "spoken_form_false_start_examples": """- Example: "Poderíamos lançar em março" → "Poderíamos fazer, bom, poderíamos lançar em março"
   - Example: "O design azul" → "O azu-, o design vermelho\"""",
        "spoken_form_prolongation_examples": """- Example: "Bom isso está certo" → "Boom isso está certo"
   - Example: "Eu quero dizer" → "Eu quero diizer\"""",
    },
    "ko": {
        "code": "ko",
        "name": "Korean",
        "language_instruction": "IMPORTANT: Generate ALL text content in Korean. Use culturally appropriate Korean names, company names, and conventions. The workplace is set in South Korea.",
        "content_language_instruction": "Generate all text content (dialogue, descriptions, emails, documents) in Korean. JSON keys and structural field names remain in English.",
        # Company / location
        "location_format": "City, South Korea (e.g., Seoul, South Korea)",
        "default_location_examples": "서울; 부산; 인천; 대전; 대구; 판교",
        # Accents
        "accent_examples": [
            "서울 표준어",
            "부산 사투리",
            "경상도 억양",
            "전라도 억양",
            "충청도 억양",
        ],
        # Disfluency
        "filler_words": ["음", "어", "그", "이제", "뭐", "좀", "약간", "그래서", "아", "그러니까", "이게", "저기"],
        "hedge_words": ["아마", "아마도", "제 생각에는", "글쎄요", "좀", "어쩌면", "대략", "~인 것 같아요"],
        "backchannel_phrases": ["네", "예", "아", "맞아요", "그렇죠", "응", "그래요", "알겠습니다", "그렇군요", "맞아", "진짜요", "정말요", "오", "아하", "그래그래"],
        # Speaking rate
        "words_per_minute_range": "200-250",
        # Email conventions
        "email_greeting_formal": "[Name]님께,",
        "email_greeting_informal": "[Name]님,",
        "email_signoff_formal": "감사합니다.",
        "email_signoff_informal": "감사합니다,",
        "email_conventions_note": "Use Korean email conventions: formal greetings (님께, 귀하) for external contacts, informal (님) for colleagues. Sign-offs: 감사합니다 / 수고하셨습니다 (formal), 감사합니다 / 수고하세요 (informal). Korean workplace emails tend to use honorific speech (존댓말).",
        # Working days
        "working_days_label": "월요일부터 금요일까지",
        # Spoken form
        "spoken_form_rules": """1. **Currency**: "₩4,110" → "사천백십 원", "$50" → "오십 달러"
2. **Percentages**: "34%" → "삼십사 퍼센트", "40%" → "사십 퍼센트"
3. **Numbers in context**: spell out naturally in Korean. "200에서 800" → "이백에서 팔백". "오전 7시" → "오전 일곱 시". Keep numbers that are clearly spoken as digits (e.g. phone numbers) in a speakable form.
4. **Acronyms / initialisms**: leave well-known ones (CEO, API, IT, AI) as-is. For uncommon ones, insert periods so TTS spells them out.
5. **Backchannels**: "음." → "음", "네네" → "네네\"""",
        "spoken_form_filler_examples": """- Example: "예산을 확정해야 합니다" → "예산을, 음, 확정해야 합니다"
   - Example: "좋은 지적이에요" → "그, 좋은 지적이에요\"""",
        "spoken_form_repetition_examples": """- Example: "제 생각에는 우리가" → "제, 제 생각에는 우리가"
   - Example: "일정이 좀" → "일정이, 일정이 좀\"""",
        "spoken_form_false_start_examples": """- Example: "3월에 출시할 수 있을 것 같아요" → "3월에 할 수, 아 그러니까, 3월에 출시할 수 있을 것 같아요"
   - Example: "파란색 디자인이" → "파란, 아니 빨간색 디자인이\"""",
        "spoken_form_prolongation_examples": """- Example: "그래서 괜찮아요" → "그래서어 괜찮아요"
   - Example: "제 말은" → "제에 말은\"""",
    },
}


def get_lang_config(language_code: str) -> dict:
    """Return language config for the given code, defaulting to English."""
    return LANG_CONFIG.get(language_code, LANG_CONFIG["en"])
