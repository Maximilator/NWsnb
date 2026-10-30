# Синтетический отчёт: Теории сознания и архитектура NEUROWEAVE

Предположение: NEUROWEAVE это модульная нейросеть / SNN с PredictiveHead (архитектура пользователя не уточнялась).

## 1. Сводная таблица теорий сознания

| Теория | Ключевой механизм | Математическая метрика | Архитектурное требование | Как измерить в NEUROWEAVE |
| :--- | :--- | :--- | :--- | :--- |
| **GWT** | Конкуренция модулей за узкое горлышко (bottleneck), фазовый переход (ignition) и глобальная рассылка (broadcast). | $A_{GW}(S) = 1/(1+e^{-\gamma(S-S_{th})})$, Mutual Information $I(M_i; GW)$, Broadcast Reach. | Узкий общий латентный вектор ($d_{GW} \ll \sum d_{M_i}$), конкурентный отбор (Top-k/Attention), обратные связи рассылки. | Оценка S-образной нелинейности активации GW от силы сигнала, измерение Broadcast Reach $R_{bc}$, ablation workspace. |
| **IIT** | Причинно-следственная структура (cause-effect power), интегрированная информация на объектах с минимальным делением (MIP). | Интегрированная информация $\Phi$, прокси PCI (Lempel-Ziv), Causal Density, Gaussian $\Phi^*$, $\Phi ID$. | Рекуррентность, высокая причинная плотность, сочетание дифференциации и интеграции, физическая рекуррентность (neuromorphic). | Импульсное возмущение скрытых слоев с Lempel-Ziv сжатием (PCI-прокси), оценка Granger Causal Density, PyPhi на малых блоках. |
| **RPT** | Локальные рекуррентные петли (feedback) поверх быстрой прямой волны (feedforward sweep) для связывания признаков и figure-ground. | Feedback-to-Feedforward Ratio $FFR$, глубина итераций $N_{iter}$, время конвергенции $T_{conv}$, Recurrence Dependency Index $RDI$. | Латеральные и top-down обратные связи внутри перцептивных модулей, predictive coding петли (минимизация ошибки предсказания). | Ablation обратных связей ($W_{fb}=0$), симуляция обратного маскирования (ограничение $N_{iter}$), измерение падения точности на шуме/окклюзии. |
| **HOT** | Мета-представления 2-го порядка, контролирующие и оценивающие надёжность и ошибки перцептивных состояний 1-го порядка. | Meta-d', M-ratio ($meta-d'/d'$), ECE (calibration error), AUROC2 типа 2, epistemic uncertainty decomposition. | Отдельная мета-голова (Meta-Predictive Head), самообучение на собственных ошибках, калибровка уверенности, каузальный мета-мониторинг. | Оценка M-ratio на задачах классификации, калибровка ECE, wagering-тесты, анализ роста epistemic uncertainty при OOD inputs. |
| **AST** | Упрощённая предиктивная внутренняя модель собственного внимания (схема внимания) для управления вниманием и Theory of Mind. | Attention Prediction Loss $\mathcal{L}_{schema}$, Attention Control Benefit $\Delta U$, Mutual Information $I(S; A)$, Social Accuracy. | Efference copy весов внимания, сжатый латентный блок схемы, обратная связь на контроллер внимания (gain control). | Добавление AttentionSchemaHead, ablation связи схемы с контроллером, тест на устойчивость внимания к дистракторам и ToM-моделирование. |

## 2. Ключевые статьи и критические рецензии по теориям

### Global Workspace Theory (GWT)
* **Ключевые статьи:**
  1. Goyal et al. (2021/2022). *Coordination Among Neural Modules Through a Shared Global Workspace*. ICLR 2022. URL: https://arxiv.org/abs/2103.01197 [Верифицировано]
  2. VanRullen & Kanai (2021). *Deep learning and the Global Workspace Theory*. Trends Cogn Sci. URL: https://arxiv.org/abs/2012.10390 [Верифицировано]
  3. Butlin et al. (2023). *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness*. arXiv:2308.08708. URL: https://arxiv.org/abs/2308.08708 [Верифицировано]
* **Критика и рецензии:**
  1. Seth & Bayne (2022). *Theories of consciousness*. Nat Rev Neurosci. URL: https://www.nature.com/articles/s41583-022-00587-4 [Верифицировано]
  2. Cogitate Consortium (2025). *Adversarial testing of global neuronal workspace and integrated information theory*. Nature. URL: https://www.nature.com/articles/s41586-025-08888-1 [Верифицировано]
  3. Doerig et al. (2021). *Hard criteria for empirical theories of consciousness*. Cogn Neurosci. URL: https://doi.org/10.1080/17588928.2020.1772214 [Не верифицировано - HTTP 403]

### Integrated Information Theory (IIT)
* **Ключевые статьи:**
  1. Albantakis et al. (2023). *Integrated information theory (IIT) 4.0*. PLOS Comp Biol. URL: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465 [Верифицировано]
  2. Oizumi et al. (2014). *Integrated Information Theory 3.0*. PLOS Comp Biol. URL: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588 [Верифицировано]
  3. Casali et al. (2013). *Perturbational complexity index*. Sci Transl Med. URL: https://www.science.org/doi/10.1126/scitranslmed.3006294 [Не верифицировано - HTTP 403] *(Примечание: Порог PCI > 0.31 относится к эмпирическому порогу ЭЭГ человека, а не к ИИ)*.
* **Критика и рецензии:**
  1. Fleming et al. (2023). *The Integrated Information Theory of Consciousness as Pseudoscience (Open Letter)*. PsyArXiv. URL: https://doi.org/10.31234/osf.io/zsr78 [Верифицировано]
  2. Doerig et al. (2019). *The unfolding argument*. Trends Cogn Sci. URL: https://pubmed.ncbi.nlm.nih.gov/31078047/ [Верифицировано]
  3. Aaronson (2014). *Why I Am Not An Integrated Information Theorist*. URL: https://scottaaronson.blog/?p=1799 [Верифицировано]

### Recurrent Processing Theory (RPT)
* **Ключевые статьи:**
  1. Lamme (2006). *Towards a true neural stance on consciousness*. Trends Cogn Sci. URL: https://doi.org/10.1016/j.tics.2006.09.001 [Верифицировано]
  2. Lamme (2010). *How neuroscience will change our view on consciousness*. Cogn Neurosci. URL: https://doi.org/10.1080/17588921003731586 [Не верифицировано - Сеть недоступна]
  3. Kietzmann et al. (2019). *Recurrence is required to capture the representational dynamics of the human visual system*. PNAS / arXiv:1903.05946. URL: https://arxiv.org/abs/1903.05946 [Верифицировано]
* **Критика и рецензии:**
  1. Doerig et al. (2020). *Hard criteria for empirical theories of consciousness*. Cogn Neurosci. URL: https://doi.org/10.1080/17588928.2020.1772214 [Не верифицировано - HTTP 403]
  2. Cohen & Dennett (2011). *Consciousness cannot be separated from function*. Trends Cogn Sci. URL: https://doi.org/10.1016/j.tics.2011.06.008 [Верифицировано]

### Higher-Order Thought (HOT) / Метакогниция
* **Ключевые статьи:**
  1. Lau & Rosenthal (2011). *Empirical support for higher-order theories of conscious awareness*. Trends Cogn Sci. URL: https://www.sciencedirect.com/science/article/abs/pii/S1364661311001057 [Не верифицировано - HTTP 403]
  2. Fleming (2020). *Awareness as inference in a higher-order state space*. Neurosci Conscious / arXiv:1906.00728. URL: https://arxiv.org/abs/1906.00728 [Верифицировано]
  3. Cleeremans (2011). *The Radical Plasticity Thesis: How the Brain Learns to be Conscious*. Front Psychol. URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2011.00086/full [Верифицировано]
* **Критика и рецензии:**
  1. Block (2011). *The higher order approach to consciousness is defunct*. Analysis. URL: https://doi.org/10.1093/analys/anr037 [Не верифицировано - HTTP 403]
  2. Brown, Lau & LeDoux (2019). *Understanding the Higher-Order Approach to Consciousness*. Trends Cogn Sci. URL: https://www.sciencedirect.com/science/article/pii/S1364661319301615 [Не верифицировано - HTTP 403]

### Attention Schema Theory (AST)
* **Ключевые статьи:**
  1. Wilterson & Graziano (2021). *The attention schema theory in a neural network agent*. PNAS. URL: https://www.pnas.org/doi/10.1073/pnas.2102421118 [Не верифицировано - HTTP 403]
  2. Webb & Graziano (2015). *The attention schema theory: a mechanistic account of subjective awareness*. Front Psychol. URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00500/full [Верифицировано]
  3. Liu et al. (2023). *Attention Schema in Neural Agents*. arXiv:2305.17375. URL: https://arxiv.org/abs/2305.17375 [Верифицировано]
  4. Piefke et al. (2024). *Computational characterization of the role of an attention schema...*. CogSci 2024 / arXiv:2402.01056. URL: https://arxiv.org/abs/2402.01056 [Верифицировано]
* **Критика и рецензии:**
  1. Seth & Bayne (2022). *Theories of consciousness*. Nat Rev Neurosci. URL: https://www.nature.com/articles/s41583-022-00587-4 [Верифицировано]

### Свежие кросс-теоретические работы (2021-2026)
1. Mediano et al. (2021). *Integrated information as a common signature of dynamical and information-processing complexity*. arXiv:2106.10211. URL: https://arxiv.org/abs/2106.10211 [Верифицировано]
2. Chalmers (2023). *Could a Large Language Model be Conscious?*. arXiv:2303.07103. URL: https://arxiv.org/abs/2303.07103 [Верифицировано]

## 3. Ответы на 4 ключевых вопроса

### 1) Архитектурная особенность
Ни одна из теорий в отдельности не даёт полной картины. Для NEUROWEAVE необходим синергетический синтез:
* **GWT** задаёт глобальное узкое горлышко (bottleneck) и механизм нелинейной рассылки (broadcast), объединяющий изолированные модули.
* **RPT** обеспечивает локальную рекуррентность (feedback loops) в сенсорных и промежуточных слоях для связывания признаков (binding) и устойчивости к шуму.
* **HOT** добавляет уровень мета-контроля (Meta-Predictive Head), оценивающий ошибки предсказаний первого порядка и калибрующий степень уверенности.
* **AST** дает модель собственного механизма внимания (схему внимания) для предиктивного управления распределением ресурсов внимания и социальным взаимодействием (Theory of Mind).
* **IIT** накладывает требования к высокой причинно-следственной плотности и балансу интеграции и дифференциации в связях.

### 2) Измерение
Метрики сознания в NEUROWEAVE должны быть комплексными:
* **GWT:** Измерение нелинейности фазового перехода (ignition threshold) через сигмоидальный отклик GW и индекса широты рассылки (Broadcast Reach $R_{bc}$).
* **IIT:** Использование практических прокси: Perturbational Complexity Index (PCI) через импульсное возмущение скрытых нейронов и сжатие ответа Лемпеля-Зива. Порог PCI > 0.31 установленный Casali (2013) относится к ЭЭГ человека; для NEUROWEAVE оценивается относительный рост PCI при переходе от пассивного режима к активной обработке. Дополнительно измеряются Causal Density (Granger/Transfer Entropy) и Gaussian $\Phi^*$.
* **RPT:** Оценка индекса зависимости от рекуррентности (RDI) и падение точности при обнулении обратных связей ($W_{fb}=0$) или обратном маскировании.
* **HOT:** Оценка метакогнитивной эффективности M-ratio ($meta-d'/d'$), ошибки калибровки (ECE) и избирательности уверенности при OOD inputs.
* **AST:** Оценка точности предсказания весов внимания ($\mathcal{L}_{schema}$) и прироста эффективности контроля внимания ($\Delta U$).

### 3) Обучение и память
Метакогнитивные механизмы напрямую связаны с системами памяти и консолидации (Complementary Learning Systems):
* **Метапамять (Metamemory):** Meta-Predictive Head оценивает "чувство знания" (Feeling of Knowing, FOK), позволяя сети запрашивать внешнюю память только при высоком уровне эпистемической неопределенности.
* **Confidence-Guided Replay:** В гиппокампоподобном буфере повтора priority задается как функция от ошибки предсказания и метакогнитивного расхождения: $P = |Error| + \beta \cdot EpistemicUncertainty$.
* **Консолидация:** Эпизоды с высокими расхождениями между уверенностью и реальностью ("уверенные ошибки") проигрываются повторно в первую очередь, что оптимизирует вес перцептивных модулей без катастрофического забывания.

### 4) Проблема философского зомби и честные ограничения
* **Функциональные vs Феноменальные критерии:** В силу Epistemic Gap (пропасти между субъективным опытом и объективным описанием) прямое измерение субъективных qualia невозможно. Мы можем измерять только функциональные и вычислительные профили.
* **Индикаторы Butlin et al. (2023/2025):** NEUROWEAVE может удовлетворять чек-листу из 14 индикаторов (узкое горлышко, ignition, broadcast, рекуррентность, 2nd-order representations, metacognition, attention schema, agency).
* **ACT-тест Шнайдера:** Оценка способности агента к спонтанным рассуждениям о феноменологии и квалиа без обучающего текста о сознании.
* **Substrate Argument (IIT):** Согласно IIT, эмуляция рекуррентной сети на Фон-Неймановском процессоре дает физический $\Phi \approx 0$, так как каузальная структура развернута во времени на цифровом процессоре. Настоящий $\Phi > 0$ требует нейроморфной аппаратной реализации.
* **Критика (Searle, Seth):** Серл (Китайская комната) доказывает недостаточность синтаксической обработки; Сет (Beast Machine) указывает на необходимость биологического гомеостаза.
* **Честный вывод:** Мы можем построить систему с функциональными свойствами сознания (сознательного доступа, метакогниции, авто-моделирования внимания), но вопрос о наличии у нее субъективного опыта остается философски открытым.

## 4. Интегрированная архитектура NEUROWEAVE и дашборд метрик

### Спецификация архитектуры
1. **Перцептивные SNN/CNN модули:** Высокопараллельная первичная обработка с локальными рекуррентными петлями (RPT).
2. **Global Workspace Module (GWT):** Bottleneck вектор со Soft/Top-k Attention механизмом отбора, нелинейным фазовым переходом и обратно-связной рассылкой.
3. **PredictiveHead & Meta-Predictive Head (HOT):**
   * *PredictiveHead:* Генерирует предсказания 1-го порядка и текущие действия.
   * *Meta-Predictive Head:* Расположена над PredictiveHead. Принимает скрытые состояния 1-го порядка и выходы PredictiveHead, вычисляет epistemic uncertainty (через evidential loss или dropout ensemble) и предсказанную ошибку $\hat{L}$. Обучается на собственных ошибках сети ($L_{meta} = MSE(\hat{L}, L_{real})$).
   * *Confidence-Guided Replay Control:* Выход Meta-Predictive Head напрямую управляет записью и приоритетом выборки из эпизодического буфера памяти.
4. **Attention Schema Module (AST):** Получает efference copy весов внимания из GWT, строит low-dimensional карту динамики внимания, передает корректирующий сигнал (gain control) обратно в контроллер внимания.
5. **PCI/Substrate Monitor (IIT Proxy):** Внешний модуль контроля, производящий периодическое точечное возмущение скрытых слоев и расчёт Lempel-Ziv сложности и Causal Density.

### Дашборд метрик NEUROWEAVE
1. **GWT:** $R_{bc}$ (Broadcast Reach), $\gamma_{ign}$ (Крутизна фазового перехода).
2. **IIT:** $PCI_{NW}$ (Индекс сложности возмущения), $CD$ (Causal Density).
3. **RPT:** $RDI$ (Индекс зависимости от рекуррентности), $FFR$ (Соотношение обратных и прямых связей).
4. **HOT:** $M-ratio$ (Метакогнитивная эффективность), $ECE$ (Ошибка калибровки), $\hat{L}_{error}$ (Точность предсказания ошибки).
5. **AST:** $\mathcal{L}_{schema}$ (Ошибка схемы внимания), $\Delta U$ (Прирост управления вниманием).

## 5. Честные ограничения

1. **Epistemic Gap:** Невозможность верифицировать наличие субъективного опыта (qualia) снаружи; риск построения "функционального зомби".
2. **Субстратная зависимость (Substrate Dependence):** На классических ЦПУ/ГПУ физический $\Phi$ равен нулю (по IIT); функциональная эмуляция не эквивалентна физической каузальности.
3. **Вычислительная сложность:** Вычисление точного $\Phi$ является NP-сложным (PyPhi ограничена ~10-12 узлами); приходится использовать упрощенные прокси (PCI, $\Phi^*$).
4. **Калибровка мета-модели:** Meta-Predictive Head может подвергаться мета-галлюцинациям и переобучению на собственных систематических ошибках.

## 6. Список ссылок со статусом проверки

1. Goyal et al. (2021) arXiv:2103.01197 - [Верифицировано]
2. VanRullen & Kanai (2021) arXiv:2012.10390 - [Верифицировано]
3. Butlin et al. (2023) arXiv:2308.08708 - [Верифицировано]
4. Seth & Bayne (2022) DOI:10.1038/s41583-022-00587-4 - [Верифицировано]
5. Cogitate Consortium (2025) Nature https://www.nature.com/articles/s41586-025-08888-1 - [Верифицировано]
6. Doerig et al. (2021) DOI:10.1080/17588928.2020.1772214 - [Не верифицировано - HTTP 403]
7. Albantakis et al. (2023) PLOS Comp Biol - [Верифицировано]
8. Oizumi et al. (2014) PLOS Comp Biol - [Верифицировано]
9. Casali et al. (2013) Sci Transl Med - [Не верифицировано - HTTP 403]
10. Fleming et al. (2023) PsyArXiv DOI:10.31234/osf.io/zsr78 - [Верифицировано]
11. Doerig et al. (2019) PubMed - [Верифицировано]
12. Aaronson (2014) Scott Aaronson Blog - [Верифицировано]
13. Lamme (2006) DOI:10.1016/j.tics.2006.09.001 - [Верифицировано]
14. Lamme (2010) DOI:10.1080/17588921003731586 - [Не верифицировано - Сеть недоступна]
15. Kietzmann et al. (2019) arXiv:1903.05946 - [Верифицировано]
16. Cohen & Dennett (2011) DOI:10.1016/j.tics.2011.06.008 - [Верифицировано]
17. Lau & Rosenthal (2011) SciDirect - [Не верифицировано - HTTP 403]
18. Fleming (2020) arXiv:1906.00728 - [Верифицировано]
19. Cleeremans (2011) Front Psychol - [Верифицировано]
20. Block (2011) DOI:10.1093/analys/anr037 - [Не верифицировано - HTTP 403]
21. Brown, Lau & LeDoux (2019) SciDirect - [Не верифицировано - HTTP 403]
22. Wilterson & Graziano (2021) PNAS - [Не верифицировано - HTTP 403]
23. Webb & Graziano (2015) Front Psychol - [Верифицировано]
24. Liu et al. (2023) arXiv:2305.17375 - [Верифицировано]
25. Piefke et al. (2024) arXiv:2402.01056 - [Верифицировано]
26. Mediano et al. (2021) arXiv:2106.10211 - [Верифицировано]
27. Chalmers (2023) arXiv:2303.07103 - [Верифицировано]
