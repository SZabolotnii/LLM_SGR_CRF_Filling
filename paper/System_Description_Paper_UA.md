# MedGemma StructCore: Schema-Guided Condensation and Deterministic Compilation for CRF Filling (CL4Health 2026, системний опис)

**Мова цього драфту:** українська (MD). Після верифікації — перенос у LaTeX англійською.  
**Автори:** [Команда / Автори]  
**Версія:** 2026-02-25 (узгоджено з `Experiments/dyspnea_crf_eval/README.md` і `Experiments/dyspnea_crf_eval/PROGRESS_REPORT_UA.md`)

## Анотація
Заповнення Case Report Forms (CRF) з неструктурованих клінічних нотаток є практично важливим, але технічно складним завданням через шум, неоднозначність та високий ризик «галюцинацій» моделей. У системному описі ми представляємо **MedGemma StructCore** — локально-орієнтований двостадійний пайплайн для **CRF:filling Shared Task (CL4Health 2026)** на датасеті Dyspnea CRF (134 елементи).

Ключова особливість цього Shared Task — **екстремальна розрідженість**: значна частка полів має значення `unknown`, а офіційний скоринг карає «порожні» значення та хибні позитивні заповнення. У відповідь на це ми змінили підхід «одна LLM → одразу 134 поля» на **контракт-орієнтований** ланцюжок:

`Clinical Note → Stage1 (SGR-style JSON summary, 9 ключів) → Stage2 (детерміністичний компілятор + нормалізація vocab) → Submission.jsonl`

Наша ключова інженерна ідея — зробити Stage2 **0-LLM** і повністю відтворюваним: ми компілюємо офіційні 134 predictions зі Stage1 summary, нормалізуємо значення до controlled vocabulary та заповнюємо відсутнє як `unknown`. Додатково ми інтегрували **UMLS alias mapping** (покриття 134/134 item names) як опційний режим канонізації назв полів, а також evidence-gated фільтри для high-FP пунктів.

## 1. Вступ
Shared Task CL4Health 2026 з CRF-filling моделює практичний сценарій: клінічний наратив потрібно перетворити в строго визначену форму з 134 елементів, причому частина значень — категоріальні (контрольований словник), частина — числові/вимірювання, а більшість часто є «не зазначено» (`unknown`).

У таких умовах «прямий» підхід *LLM → 134 поля* нестабільний: модель або (a) «домислює» значення, збільшуючи FP, або (b) дрейфує за форматом (JSON drift), роблячи результат невалідним для Codabench. Наш підхід формалізує взаємодію з моделями через **SGR** та через **контракт** між датасетом, інтермедіатами й submission.

### 1.1 Короткий огляд Schema-Guided Reasoning (SGR)
**Schema-Guided Reasoning (SGR)** — архітектурний патерн інженерії LLM-систем, який зміщує акцент із «вільної генерації тексту» на **кероване схемою** виведення з відтворюваним результатом. У SGR експертні знання предметної області кодуються як **схеми/контракти даних**, а модель примушується дотримуватись кроків і структури через механізми *Structured Output / Constrained Decoding*. Це зменшує варіативність, знижує ризик галюцинацій і дає строго типізований вихід, який можна автоматично компілювати та валідовувати.

У нашій системі ми використовуємо SGR у Stage1 як спосіб отримати **стабільне проміжне представлення** (9 доменних ключів). Далі Stage2 детерміновано реалізує «бізнес-логіку контракту»: канонізацію ключів, нормалізацію до controlled vocabulary, evidence-gated фільтри та розгортання у повний submission (134 items, missing→`unknown`).

## 2. Датасет, формат та зміна принципу парсингу
### 2.1 Датасети та спліти
Ми працюємо з офіційними наборами Hugging Face (`NLP-FBK/dyspnea-crf-*`) та допоміжним in-domain набором `NLP-FBK/dyspnea-clinical-notes` для синтезу даних (Teacher Generation).

### 2.2 Офіційний формат submission та роль `unknown`
Офіційний submission — JSONL, де для кожного документа треба видати **всі 134 items** у фіксованому списку:
```json
{"document_id":"<id>_en","predictions":[{"item":"...","prediction":"..."}, ...]}
```
Критично: пропуски/порожні значення погіршують якість так само або сильніше, ніж явне `unknown`. Тому в нашому пайплайні **missing завжди детерміновано заповнюється як `unknown`** на фінальному кроці.

### 2.3 Нова «контрактна» модель парсингу (dataset → pipeline)
Ми перейшли від «ручного» формування списків полів та id до контрактного узгодження з самим датасетом:

1. **Список items** інфериться з офіційного record (`annotations[*].item` або `expected_crf_items`) і використовується як єдиний source of truth для порядку та повноти.
2. **Нормалізація `document_id`:** всередині пайплайна ми працюємо з `plain_id` (до першого `_`), щоб узгодити артефакти, GT і submission; для submission додаємо суфікс `_<lang>`.
3. **Sparse внутрішнє представлення:** Stage1 і Stage2 оперують «наявними фактами», але фінальна стадія завжди розгортає їх у повний список 134 items і заповнює відсутнє як `unknown`.

Ця зміна суттєво спростила відтворюваність, локальний скоринг і дебаг: ми відокремили (a) парсинг/узгодження датасету та контрактів від (b) якості extraction.

## 3. Огляд системи
### 3.1 Архітектура
```
Clinical note
  └─ Stage1: LLM → SGR-style JSON (9 ключів)
       └─ Stage1→text: детермінований короткий summary
            └─ Stage2(det): парсинг + канонізація items + нормалізація значень
                 └─ Submission builder: 134 items, missing→unknown, zip для Codabench
```

### 3.2 Stage1 контракт (profile=9)
Stage1 повертає один JSON з 9 ключами:
`DEMOGRAPHICS, VITALS, LABS, PROBLEMS, SYMPTOMS, MEDICATIONS, PROCEDURES, UTILIZATION, DISPOSITION`.

Фокус Stage1 — **висока стабільність формату** (JSON parse OK) та **sparse** стиль: включати тільки факти, які явно підтримані текстом.

**Поточний стан (Preliminary, 2026-02-24, train10):** `json_parse_ok=10/10`, `thinking_leak=0/10`.

### 3.3 Stage2(det): детерміністичний компілятор
Stage2 не викликає LLM. Він:
- розпарсує Stage1 summary,
- канонізує item keys (exact/casefold/апострофи/абревіатури; опційно UMLS aliases),
- нормалізує значення у controlled vocabulary,
- збирає повний submission-обʼєкт із 134 items (missing→`unknown`),
- застосовує FP-gates для high-FP пунктів на основі evidence.

## 4. Методологія
### 4.1 Канонізація назв item (опційно UMLS)
Ми підготували UMLS mapping для всіх 134 item names (coverage `134/134`, **Verified**) та інтегрували його як опційний режим у Stage2 runners (`--use-umls-mapping`). У цьому режимі система робить best-effort alias resolution перед фінальним exact-match.

### 4.2 Нормалізація значень до controlled vocabulary
Детерміністичний модуль приводить значення до офіційного словника (наприклад: `yes/no` → `y/n`, AVPU, chronic/neoplasia/duration, числові значення лабораторій тощо) та уніфікує варіанти `unknown`.

### 4.3 FP-gates (evidence-gated постфільтри)
Для найбільш «дорогих» за FP пунктів додані evidence-gated правила, щоб не заповнювати їх без достатньої опори у Stage1 summary (наприклад: `active neoplasia`, `arrhythmia`, `acute coronary syndrome`, `presence of dyspnea`).

## 5. Експерименти та результати (стан на 2026-02-24)
Ми звітуємо результати з чіткими етикетками надійності (через малий розмір підмножин).

### 5.1 Stage1 teacher vs student (N=10, Preliminary)
Teacher Stage1 (Gemini) + Stage2(det): `macro_f1=0.6763`, `TP/FP/FN=68/35/9`.  
Student Stage1 (MedGemma base) + Stage2(det): `macro_f1=0.6521`, `TP/FP/FN=57/13/22`.

Інтерпретація: FN у student-run переважно Stage1 recall-limited; зростання FP у teacher-run сигналізує потребу в жорсткіших FP-gates на Stage2(det) для teacher-like Stage1.

### 5.2 SGR-only end-to-end baseline (train10, Preliminary)
Stage1 (SGR JSON profile=9) → Stage2(det) показав (train=10):
- `macro_f1=0.438431`
- `TP/FP/FN=39/44/39`

Цей baseline фіксує стабільність формату та повну відтворюваність Stage2 без LLM, але потребує подальшої роботи над recall у Stage1 та над FN-first дериваціями у compiler.

## 6. Обмеження та аналіз помилок (попередньо)
- **Recall обмежений Stage1:** якщо факт не потрапив у Stage1 summary, Stage2(det) не може його «відновити» з raw note.
- **Неявні згадки:** частина CRF items може вимагати контексту поза коротким summary (ризик FN).
- **Висока ціна FP:** агресивні правила/синоніми без evidence погіршують Macro-F1, тому пріоритезується evidence-gated заповнення.

## 7. Відтворюваність
Код та артефакти експериментів організовані в `Experiments/dyspnea_crf_eval/`. Типові entrypoints:
- Stage1: `Experiments/dyspnea_crf_eval/medgemma_crf_repo/scripts/run_stage1_only_train10.py`
- Stage2(det): `Experiments/dyspnea_crf_eval/run_stage2_deterministic_from_stage1.py`
- End-to-end (train10): `Experiments/dyspnea_crf_eval/run_sgr_only_pipeline_train10.py`

Submission builder гарантує валідність структури та може запакувати ZIP з `mock_data_dev_codabench.jsonl` у корені архіву (конвенція Codabench).

## 8. Етичні міркування та відповідність
Ми використовуємо публічно доступні датасети з умовами використання організаторів. Репозиторій не містить приватних клінічних даних або сирих нотаток поза тим, що надано організаторами у відкритому доступі. Синтетичні дані (якщо використовуються для навчання) повинні супроводжуватися описом валідації та ризиків помилок/упереджень.

## 9. Висновки
MedGemma StructCore у поточній ітерації фокусується на **форматній стабільності**, **контрактному парсингу датасету** та **детерміністичному Stage2 компіляторі**. Це дозволяє відокремити проблеми extraction від проблем submission-формату й controlled vocabulary. Наступні кроки — підвищення recall Stage1 (особливо для high-FN items) та перевірка якості на більших підмножинах (dev80/test200) з фіксацією артефактів і регресійних гейтів.
