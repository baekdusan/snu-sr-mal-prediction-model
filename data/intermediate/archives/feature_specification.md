## Problem_Understanding

- Goal: design theory-grounded, interpretable features to predict the Maximum Acceptable Latency (MAL) for each natural-language query in your dataset.
- DV: `MAL` (in seconds). Unit of analysis: one row per query (`queries`).
- Available signal: only the Korean query text. We therefore focus on text-derived features: task type (lookup vs analysis vs generation), modality (photo/video/calendar/finance), urgency/time-horizon, personalization level, etc.
- Below, I define a compact **core set** plus **optional** features that can be implemented via simple NLP rules or lightweight models.

---

## Feature_Specification

**Legend**

- **Core set** features are marked with `⭐` in the name.
- All others are optional / for richer models.

| feature_name | type | derived_from | definition | theoretical_rationale |
| --- | --- | --- | --- | --- |
| ⭐`QL_chars` | numeric | `queries` | Number of characters in the query string (after trimming, excluding surrounding quotes). | Longer queries tend to encode more constraints and higher task complexity, which can increase both computation and users’ tolerance for waiting (higher MAL). |
| ⭐`QL_words` | numeric | `queries` | Number of whitespace-separated tokens (or morpheme-count if using a Korean tokenizer). | More words typically indicate more detailed or multi-part requests, associated with higher mental model of “complex task,” which increases acceptable latency. |
| ⭐`task_category` | categorical | `queries` | Manually or rule-based classify each query into high-level task types: e.g. `{retrieve_personal_log, retrieve_media, summarize_text, generate_list, compare_stats, recommend_content, recommend_place, recommend_product, aggregate_stats, create_video, other}`. | Task type strongly shapes expected system effort and output richness; users tolerate more latency for summarization, aggregation, or media creation than for simple retrieval or a single recommendation. |
| ⭐`modality_type` | categorical | `queries` | Label dominant data modality referenced: `{none/implicit, text_note, photo_image, video, audio/music, calendar/schedule, finance/transactions, location/places, health/fitness, shopping/commerce, messaging/chat}`. Use keyword rules (e.g., “사진/영상/녹음/회의록/메모/강의노트/카드/결제/캘린더/위치” etc.). | Queries involving large or complex modalities (video, many photos, transaction logs) convey higher computational load and richer output expectations, typically increasing acceptable latency. |
| ⭐`temporal_scope_level` | ordinal | `queries` | Ordinal scale of how broad the time range is: e.g. `1=momentary (어제 점심/이번 주 하루)`, `2=day/week-level`, `3=month-level (이번 달/지난달)`, `4=multi-month/season/year (올해/지난해/작년 겨울 등)`, based on detected time expressions. | Broader temporal scopes imply more data to scan/aggregate; users expect the system to “dig deeper,” often accepting higher latency than for very narrow, recent events. |
| ⭐`urgency_level` | ordinal | `queries` | Ordinal sense of immediacy: `1=low` (retrospective analytics, comparisons, lists), `2=medium` (planning, generic recommendations), `3=high` (today/now/현재 위치에서/지금 여기/오늘 약속·회의·미팅 등). Rule-based from time words and context. | Under higher time pressure or near-term decision needs, users are less tolerant of delay (lower MAL); urgency is a robust predictor of acceptable waiting time. |
| ⭐`personalization_depth` | ordinal | `queries` | Ordinal level of personalization: `0=generic (no “내/나/내가/내일 일정 참고” etc.)`, `1=light personal context (uses personal pronouns but not deep logs)`, `2=deep personal data mining (explicitly refers to my past behavior, logs, “내 건강 데이터/내가 자주 먹는/지난달 카드 결제 내역” etc.). | More deeply personalized queries imply the system is mining and integrating large, user-specific histories, which users expect to take longer and thus accept higher latency. |
| ⭐`requires_aggregation` | binary | `queries` | 1 if query asks to “정리해서, 요약해줘, 리스트 만들어줘, 순서대로 알려줘, 비교해줘, 분석해서, 모아서, 영상으로 만들어줘” etc.; else 0. | Aggregation/summarization/comparison tasks involve higher cognitive and computational effort, so users are more tolerant of latency versus simple lookups. |
| ⭐`requires_generation` | binary | `queries` | 1 if query primarily asks for generative content or recommendations (e.g., “추천해줘”, “코스 추천해줘”, “스타일 추천해줘”); 0 otherwise. | Generative tasks are perceived as “creative” or heavy AI work, which increases perceived effort and acceptable waiting time, but still less than complex multi-log summarization. |
| ⭐`requires_historical_search` | binary | `queries` | 1 if query references past periods/events (“지난주/지난달/지난해/작년/봄에/여름에/여행 때/캠핑장/갔던/찍어둔” etc.); 0 otherwise. | Searching across historical logs is perceived as scanning large archives; users expect this to take longer and accept higher MAL compared to using only current context. |
| ⭐`output_cardinality_expectation` | ordinal | `queries` | Ordinal estimate of how many items the user expects: `1=single item` (specific date/장소/금액/사진 한 장), `2=few items` (a short list: “몇 번이나”, “비교해줘”), `3=many items` (lists, “리스트 만들어줘”, “모두 정리해서”, “영상으로 만들어줘”, “순서대로 알려줘”). | Anticipated output size correlates with mental model of processing effort; larger expected outputs make users more tolerant of latency. |
| ⭐`stakes_importance_level` | ordinal | `queries` | Subjective stakes: `1=low (entertainment, general content)`, `2=medium (shopping, diet, exercise planning)`, `3=high (돈/결제/가계부/카드/여권/증명사진/중요 일정 등)`. Rule-based from finance/ID/schedule keywords. | Higher stakes can either reduce tolerance (need fast answer) or increase tolerance (willing to wait for accuracy). For everyday personal assistants, users often accept slightly longer delays for financial/identity tasks due to importance. |
| ⭐`social_context_present` | binary | `queries` | 1 if query references others explicitly (names or roles: “친구들/가족/팀원/동료/재희/나연이/성진이/두산이/소연이/성훈이/부모님/반려동물” etc.); 0 otherwise. | Socially embedded tasks (planning with others, summarizing group chats) often feel more complex and important, nudging MAL upwards, especially when they imply coordination. |
| ⭐`device_context_implicit` | categorical | `queries` | Categorical guess of implicit device/context: `{mobile_on_the_go, home_planning, work/productivity, shopping_context, entertainment_context, unknown}` derived from cues like “지금 여기/현재 위치/출퇴근 시간/오늘 회의/캘린더/마트/아울렛/쇼핑리스트/업무노트/회의록” etc. | On-the-go/mobile contexts and time-pressured usage usually reduce acceptable latency, while home/planning or work-analytics contexts increase tolerance. |
| ⭐`comparison_task` | binary | `queries` | 1 if query explicitly compares two or more periods/entities (“비교해줘”, “이번 달과 지난달”, “온라인 vs 오프라인”, “평일과 주말”); 0 otherwise. | Comparisons require retrieving and aligning multiple data slices, which users see as more involved than single-period stats, increasing MAL. |
| ⭐`list_ordering_required` | binary | `queries` | 1 if query asks for ordering/ranking (“순서대로 알려줘”, “가장 많이 ~ 순서대로”, “리스트 만들어줘” with implied ranking); 0 otherwise. | Ranking suggests more structured processing (sorting, scoring many items), so users expect more effort and may tolerate longer delays. |
| ⭐`media_transformation` | binary | `queries` | 1 if query asks to transform media (e.g., “영상으로 만들어줘”, “요약 정리해줘” for long notes/audio, “사진 모아서”); 0 otherwise. | Media transformation (especially creating videos or summarizing long recordings) is perceived as heavy computation, strongly associated with higher acceptable latency. |
| ⭐`calendar_or_schedule_related` | binary | `queries` | 1 if query mentions 일정, 약속, 회의, 세미나, 미팅, 캘린더, 생일, 기념일 etc.; 0 otherwise. | Time-critical schedule queries often need quick answers but can also involve log search; capturing them separately allows the model to learn whether they tend to have lower MAL due to urgency. |
| ⭐`finance_or_spending_related` | binary | `queries` | 1 if query includes finance keywords (카드, 결제, 가계부, 관리비, 저축, 용돈, 지출, 금액, 쿠폰, 할인, 최저가, 특가, 할인가 등); 0 otherwise. | Money-related queries are higher stakes and often involve log aggregation; users may both expect accuracy and accept more latency than for casual content recommendations. |
| ⭐`health_fitness_related` | binary | `queries` | 1 if query mentions 운동, 칼로리, 걸음 수, 활동량, 캠핑/나들이/산책 (as activity logs), 수면 분석, 건강 데이터 etc.; 0 otherwise. | Health/fitness analytics often involve time-series aggregation; people accept moderate latency but may be more tolerant when it feels like data analysis rather than simple lookup. |
| ⭐`shopping_or_commerce_related` | binary | `queries` | 1 if query mentions 쇼핑, 마트, 편의점, 다이소, 올리브영, 쿠팡, 이마트, 브런치 카페, 펜션, 구독 서비스, 최저가, 구매할 수 있는 곳, etc.; 0 otherwise. | Commerce tasks often involve external search and recommendation over many options, which users expect to be somewhat slow and accept moderate-to-high MAL. |
| ⭐`entertainment_media_related` | binary | `queries` | 1 if query is about 음악, 노래, 유튜브, 넷플릭스, 드라마, 영화, 게임, 팟캐스트, SNS/커뮤니티, 뉴스/트렌드/잡지/블로그 소비 등; 0 otherwise. | Entertainment queries are generally lower stakes; users may expect snappy responses, but for complex personalized recommendations they may tolerate more delay. |
| ⭐`requires_external_like_search` | binary | `queries` | 1 if query semantically resembles web/content search (find similar items/places, “비슷한 느낌의 ~ 추천해줘”, “관련된 ~ 찾아줘”, “구경해 볼 수 있는 곳 찾아줘”); 0 otherwise. | External-like search implies large search spaces and ranking, raising expected computation and MAL compared to purely local log queries. |
| ⭐`multi_source_integration` | binary | `queries` | 1 if query explicitly references combining multiple internal sources (e.g., “내 건강 데이터/내 일정/내 목표 칼로리와 운동량/내 수면 분석 데이터를 참고해서”, “재희랑 했던 대화를 분석해서”); 0 otherwise. | Integrating heterogeneous data sources is cognitively and computationally complex; users will see this as “heavy AI work” and accept higher latency. |
| `contains_explicit_time_budget` | binary | `queries` | 1 if query mentions explicit time budget for content consumption (e.g., “지금 30분 정도 여유 시간에 볼만한 ~”, “출퇴근 시간에 볼 만한”); 0 otherwise. | When users reveal their own time budget, they may implicitly accept somewhat slower responses (since they plan ahead), but they also signal time-awareness; the model can learn the direction from data. |
| `time_reference_recency` | ordinal | `queries` | Ordinal recency of reference event: `1=very recent (오늘/어제/이번 주/이번 달)`, `2=recent past (지난주/지난달)`, `3=distant (작년/지난해/작년 겨울/봄에 갔던 등)`, `4=unspecified`. | More distant references suggest larger log search and more uncertainty, which users may expect to take longer, increasing MAL. |
| `explicit_counting_or_frequency` | binary | `queries` | 1 if query asks “몇 번이나”, “횟수”, “총 얼마나 썼지”, “총 몇 시간 재생했지” etc.; 0 otherwise. | Counting/frequency queries require scanning and summing events, more intensive than single lookup, increasing acceptable latency. |
| `explicit_comparison_target_num` | numeric | `queries` | Number of distinct periods/entities being compared, inferred from patterns like “A와 B 비교”, “이번 달과 지난달”, “평일과 주말”, “온라인 쇼핑과 오프라인 쇼핑” (e.g., 2 in these cases). | More comparison targets increase mental and computational load; users anticipate more work and may tolerate longer waits. |
| `question_specificity_level` | ordinal | `queries` | `1=very specific (exact date/time/place/card, e.g., “몇 시에”, “어느 카드로”, “어디서”)`, `2=moderately specific (type/category, e.g., “어디였지?”, “어떤 곳”)`, `3=general/analytic (patterns, lists, “가장 많이 ~”, “분석해서”)`. | Highly specific questions feel like simple lookups (lower MAL), while general/analytic questions feel like analysis (higher MAL). |
| `output_format_structured` | binary | `queries` | 1 if query implies structured/tabular output (“가계부로 작성해줘”, “리스트 만들어줘”, “정리해서”, “한 장으로 요약 정리해줘”); 0 otherwise. | Structured outputs suggest more systematic processing and formatting, which users associate with heavier work and higher acceptable latency. |
| `output_format_media` | binary | `queries` | 1 if output is explicitly a media artifact (영상, 리스트 of media, etc.); 0 otherwise. | Creating media artifacts (e.g., videos from photos) is perceived as computationally heavy, strongly increasing MAL. |
| `named_person_count` | numeric | `queries` | Count of distinct named individuals or roles (e.g., 지민, 소연, 성훈, 재희, 나연, 성진, 부모님, 친구들, 팀원들 등) detected via simple list or NER. | More people often implies more complex social context or more related data (e.g., multiple conversations), which can increase expected processing and tolerated latency. |
| `contains_location_context` | binary | `queries` | 1 if query includes explicit location context: “현재 위치에서”, “집 근처”, “여기 백화점/아울렛/마트/식당/카페”, “한강 산책 코스” etc.; 0 otherwise. | Real-time location-based queries are often mobile and context-sensitive; users may expect faster responses, potentially lowering MAL compared to offline analytics. |
| `contains_goal_or_target` | binary | `queries` | 1 if query includes explicit goals/targets: “목표 걸음수”, “목표 칼로리”, “근무 시간”, “학습 시간”, “작업 시간” etc.; 0 otherwise. | Goal/target framing indicates planning or self-regulation tasks, which users see as more important and “worth waiting for,” increasing MAL slightly. |
| `contains_discount_or_savings_focus` | binary | `queries` | 1 if query includes “할인, 쿠폰, 할인가, 특가, 최저가, 저렴한 가격” etc.; 0 otherwise. | Bargain-finding is a search/optimization task; users may accept longer latency to get a good deal, but also may be in-store and time-pressured; model can learn the net effect. |
| `contains_emotional_or_preference_inference` | binary | `queries` | 1 if query asks system to infer preferences from behavior or conversations: “분석해서 ~ 좋아할 만한”, “내가 자주 먹는/보는/듣는 ~ 기준으로” etc.; 0 otherwise. | Inferring preferences from history is cognitively complex, so users may expect that it takes longer and be more tolerant of latency. |
| `historical_span_complexity` | numeric | `queries` | Approximate span length in months between earliest and latest time references mentioned (e.g., “작년과 올해” → ~12 months; “이번 달과 지난달” → ~2 months; single-period → 1). Use simple mapping of Korean time phrases to month offsets. | Larger historical spans imply more data and complexity, increasing perceived computational effort and MAL. |
| `contains_relative_change_request` | binary | `queries` | 1 if query asks “얼마나 늘었지/줄었지/증가했지/변했지” etc.; 0 otherwise. | Relative change requires computing differences across periods, more complex than absolute counts; users may accept slightly higher latency. |
| `context_now_or_here` | binary | `queries` | 1 if query includes explicit “지금”, “현재”, “오늘” combined with contextual phrases like “여기”, “현재 위치에서”; 0 otherwise. | Strong “now/here” framing indicates immediate context and often higher urgency, which tends to reduce acceptable latency. |
| `contains_family_or_important_relationship` | binary | `queries` | 1 if query mentions 가족, 부모님, 결혼기념일, 생신, 친구 결혼식, 가족 여행, 가족 단체 사진 등; 0 otherwise. | Important relationships and events increase emotional stakes; users may accept longer waits for careful, comprehensive outputs (e.g., videos from family photos). |
| `media_volume_implied` | ordinal | `queries` | Rough ordinal of how many media items implied: `1=few (단일 사진/영상)`, `2=some (“사진 다 보여줘” but narrow context like 한 번의 여행)`, `3=many (“가족 단체 사진만 모두 모아서”, “지난달 카드 결제 내역”, “여행에서 내가 갔던 곳 모두 정리해서”)`. | More items to process/aggregate increases perceived and actual load, so users accept higher latency, especially for media-heavy tasks. |
| `analysis_vs_lookup_balance` | ordinal | `queries` | Heuristic: `1=mostly lookup` (simple “어디였지/언제였지/뭐였지”), `2=mixed` (lookup plus light aggregation/comparison), `3=mostly analysis` (요약, 분석, 정리, 리스트, 비교 across periods). | Captures overall cognitive complexity; analysis-heavy queries are associated with higher MAL. |

---

### Recommended Core Set (for a compact, strong baseline)

If you want a small but powerful, interpretable feature set, prioritize:

- `QL_chars`
- `QL_words`
- `task_category`
- `modality_type`
- `temporal_scope_level`
- `urgency_level`
- `personalization_depth`
- `requires_aggregation`
- `requires_generation`
- `requires_historical_search`
- `output_cardinality_expectation`
- `stakes_importance_level`
- `social_context_present`
- `device_context_implicit`
- `comparison_task`
- `list_ordering_required`
- `media_transformation`
- `calendar_or_schedule_related`
- `finance_or_spending_related`
- `shopping_or_commerce_related`
- `entertainment_media_related`
- `requires_external_like_search`
- `multi_source_integration`

These should give your MAL model a strong, theory-grounded foundation; the optional features can be added incrementally for improved nuance.