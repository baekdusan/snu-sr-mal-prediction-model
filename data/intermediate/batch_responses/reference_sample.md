1. **Expanded_Dataset_JSON**
```json
[
  {
    "queries": "지난주에 찍은 골프 스윙 영상 보여줘",
    "QL_char": 20,
    "QL_tokens": 5,
    "task_family": "retrieve_item",
    "info_source_type": "personal_content",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "video",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "media_output",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "neutral",
    "cognitive_load_estimate": 0,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 0.3,
    "embedding_axis_urgency": 0.0
  },
  {
    "queries": "지난주에 소연이가 추천했던 유튜브 영상 콘텐츠 제목이 뭐였지?",
    "QL_char": 34,
    "QL_tokens": 7,
    "task_family": "retrieve_item",
    "info_source_type": "media_history",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "video",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 1,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 2,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "neutral",
    "cognitive_load_estimate": 0,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 0.4,
    "embedding_axis_urgency": 0.0
  },
  {
    "queries": "사려고 찍어둔 와인 사진 다 보여줘",
    "QL_char": 18,
    "QL_tokens": 5,
    "task_family": "retrieve_item",
    "info_source_type": "personal_content",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "photo_image",
    "domain_category": "shopping_retail",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "batch_media_ops",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 1,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 2,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 0.8,
    "embedding_axis_urgency": 0.0
  },
  {
    "queries": "지난번 여행에서 산책할 때 찍은 사진 보여줘",
    "QL_char": 22,
    "QL_tokens": 6,
    "task_family": "retrieve_item",
    "info_source_type": "personal_content",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_specific",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "photo_image",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "batch_media_ops",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 1,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 0.9,
    "embedding_axis_urgency": 0.0
  },
  {
    "queries": "봄에 갔던 골프장 어디였지?",
    "QL_char": 16,
    "QL_tokens": 4,
    "task_family": "retrieve_item",
    "info_source_type": "location_history",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "structured_log",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "neutral",
    "cognitive_load_estimate": 0,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 0.2,
    "embedding_axis_urgency": 0.0
  },
  {
    "queries": "어제 반복해서 들은 노래의 제목과 가수명이 뭐지?",
    "QL_char": 27,
    "QL_tokens": 7,
    "task_family": "retrieve_item",
    "info_source_type": "media_history",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_specific",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "structured_log",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 1,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "neutral",
    "cognitive_load_estimate": 0,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 0.4,
    "embedding_axis_urgency": 0.1
  },
  {
    "queries": "작년 겨울에 구매한 패딩을 어느 사이트에서 샀었지?",
    "QL_char": 30,
    "QL_tokens": 8,
    "task_family": "retrieve_item",
    "info_source_type": "commerce_history",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 1,
    "modality_primary": "structured_log",
    "domain_category": "shopping_retail",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 2,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 1,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "financial_analytics",
    "embedding_axis_complexity": 0.6,
    "embedding_axis_urgency": -0.2
  },
  {
    "queries": "작년 결혼기념일 때 갔던 레스토랑 어디였지?",
    "QL_char": 24,
    "QL_tokens": 6,
    "task_family": "retrieve_item",
    "info_source_type": "location_history",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_specific",
    "time_urgency_level": 0,
    "stakes_level": 1,
    "modality_primary": "structured_log",
    "domain_category": "social_relationships",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 2,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 1,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 1,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 0.5,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "지난달에 운동 몇 번이나 했지?",
    "QL_char": 18,
    "QL_tokens": 5,
    "task_family": "count_stats",
    "info_source_type": "personal_activity_log",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "health_fitness",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 0.7,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "지난달에 유튜브를 총 몇 시간 재생했지?",
    "QL_char": 23,
    "QL_tokens": 7,
    "task_family": "count_stats",
    "info_source_type": "media_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "structured_log",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 1,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 0.7,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "올해 상반기에 옷을 몇 번이나 샀지?",
    "QL_char": 21,
    "QL_tokens": 6,
    "task_family": "count_stats",
    "info_source_type": "commerce_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 1,
    "stakes_level": 1,
    "modality_primary": "structured_log",
    "domain_category": "shopping_retail",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 1,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 2,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "budgeting",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "financial_analytics",
    "embedding_axis_complexity": 0.8,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "올해 캠핑을 몇 번이나 갔지?",
    "QL_char": 17,
    "QL_tokens": 5,
    "task_family": "count_stats",
    "info_source_type": "location_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 1,
    "stakes_level": 1,
    "modality_primary": "structured_log",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 2,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "budgeting",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 0.7,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "지난달과 비교했을 때 이번 달 운동량이 얼마나 늘었지?",
    "QL_char": 30,
    "QL_tokens": 9,
    "task_family": "compare_trend",
    "info_source_type": "personal_activity_log",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 1,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "health_fitness",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 1,
    "has_temporal_diff_phrase": 1,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 2,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "comparison",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "health_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "올해 가장 많이 들었던 음악 순서대로 알려줘",
    "QL_char": 24,
    "QL_tokens": 7,
    "task_family": "rank_order",
    "info_source_type": "media_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 1,
    "stakes_level": 0,
    "modality_primary": "structured_log",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 2,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 1,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "content_reco",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "comparison",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.2
  },
  {
    "queries": "온라인 쇼핑과 오프라인 쇼핑 결제 금액 비교해줘",
    "QL_char": 25,
    "QL_tokens": 7,
    "task_family": "compare_trend",
    "info_source_type": "commerce_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 0,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "finance",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 1,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 1,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "tabular_report",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "comparison",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "financial_analytics",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.3
  },
  {
    "queries": "여행 갔을 때 평소보다 평균 걸음 수가 얼마나 늘었지?",
    "QL_char": 30,
    "QL_tokens": 9,
    "task_family": "compare_trend",
    "info_source_type": "personal_activity_log",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_specific",
    "time_urgency_level": 0,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "health_fitness",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 1,
    "has_temporal_diff_phrase": 1,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "other",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "comparison",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "health_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "이번 주 요일 별 칼로리 소모량을 순서대로 알려줘",
    "QL_char": 26,
    "QL_tokens": 8,
    "task_family": "rank_order",
    "info_source_type": "personal_activity_log",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "present_now",
    "time_urgency_level": 2,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "health_fitness",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "other",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 1,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "comparison",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "health_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 0.5
  },
  {
    "queries": "작년 이맘때쯤 자주 들었던 음악 리스트 만들어줘",
    "QL_char": 27,
    "QL_tokens": 8,
    "task_family": "list_history",
    "info_source_type": "media_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "structured_log",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 2,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 1,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "content_reco",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "comparison",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.3
  },
  {
    "queries": "이번주 마트에서 사야하는 물건 찍어둔 것 다 모아서 쇼핑리스트 만들어줘",
    "QL_char": 39,
    "QL_tokens": 11,
    "task_family": "list_history",
    "info_source_type": "personal_content",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "present_now",
    "time_urgency_level": 2,
    "stakes_level": 1,
    "modality_primary": "photo_image",
    "domain_category": "shopping_retail",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "batch_media_ops",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "other",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 1,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 1,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "budgeting",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "personal_media_retrieval",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 0.6
  },
  {
    "queries": "스페인 여행에서 내가 갔던 곳 모두 정리해서 알려줘",
    "QL_char": 27,
    "QL_tokens": 8,
    "task_family": "list_history",
    "info_source_type": "location_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 1,
    "requires_generation": 1,
    "temporal_reference_type": "past_specific",
    "time_urgency_level": 0,
    "stakes_level": 1,
    "modality_primary": "structured_log",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 2,
    "interaction_pattern": "analytic_report",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "place_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "summary_text",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "spend_tracking",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "activity_stats",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.2
  },
  {
    "queries": "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘",
    "QL_char": 25,
    "QL_tokens": 8,
    "task_family": "recommend",
    "info_source_type": "external_web",
    "requires_personal_history": 0,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 1,
    "stakes_level": 2,
    "modality_primary": "mixed/unknown",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 1,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "place_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "health_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 0.9,
    "embedding_axis_urgency": 0.3
  },
  {
    "queries": "블랙핑크 '뛰어' 랑 비슷한 느낌의 노래 추천해줘",
    "QL_char": 28,
    "QL_tokens": 8,
    "task_family": "recommend",
    "info_source_type": "external_web",
    "requires_personal_history": 0,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "media_history",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 0,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "content_reco",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 0,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "neutral",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 1,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 0.8,
    "embedding_axis_urgency": 0.0
  },
  {
    "queries": "이 사진에 있는 신발이랑 비슷한 디자인 찾아줘",
    "QL_char": 23,
    "QL_tokens": 7,
    "task_family": "recommend",
    "info_source_type": "personal_content",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 0,
    "stakes_level": 1,
    "modality_primary": "photo_image",
    "domain_category": "fashion_style",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "batch_media_ops",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 1,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "product_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 1,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 1,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 0.2
  },
  {
    "queries": "지난달에 갔던 캠핑장과 비슷한 곳 찾아줘",
    "QL_char": 23,
    "QL_tokens": 7,
    "task_family": "recommend",
    "info_source_type": "location_history",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "past_range",
    "time_urgency_level": 0,
    "stakes_level": 1,
    "modality_primary": "mixed/unknown",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "place_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 1,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 0.9,
    "embedding_axis_urgency": 0.1
  },
  {
    "queries": "오늘 목표 걸음수를 채울 수 있는 코스 추천해줘",
    "QL_char": 25,
    "QL_tokens": 8,
    "task_family": "recommend",
    "info_source_type": "personal_activity_log",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "present_now",
    "time_urgency_level": 2,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "health_fitness",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "place_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 1,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "health_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 0.8
  },
  {
    "queries": "출퇴근 시간에 볼 만한 영상 추천해줘",
    "QL_char": 21,
    "QL_tokens": 7,
    "task_family": "recommend",
    "info_source_type": "external_web",
    "requires_personal_history": 0,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 1,
    "stakes_level": 0,
    "modality_primary": "video",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 0,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 0,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "content_reco",
    "requires_cross_app_integration": 0,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 1,
    "novelty_seeking": 1,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 0.8,
    "embedding_axis_urgency": 0.4
  },
  {
    "queries": "지금 내가 갖고 있는 멤버십 중에 여기 백화점에서 할인되는 멤버십 찾아줘",
    "QL_char": 43,
    "QL_tokens": 13,
    "task_family": "retrieve_item",
    "info_source_type": "commerce_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "present_now",
    "time_urgency_level": 2,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "finance",
    "query_goal_specificity": 1,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 1,
    "needs_health_data": 0,
    "recommendation_type": "card_coupon_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 0,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 1,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "spend_optimization",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "financial_analytics",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 1.0
  },
  {
    "queries": "현재 위치에서 걸어서 갈 수 있는 맛집 리스트 보여줘",
    "QL_char": 29,
    "QL_tokens": 9,
    "task_family": "recommend",
    "info_source_type": "external_web",
    "requires_personal_history": 0,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "present_now",
    "time_urgency_level": 2,
    "stakes_level": 1,
    "modality_primary": "mixed/unknown",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 1,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "place_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 1,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 1,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 0.9
  },
  {
    "queries": "내가 자주 먹는 음식을 분석해서 늦은 밤에 먹기 좋은 건강식 추천해줘",
    "QL_char": 38,
    "QL_tokens": 12,
    "task_family": "recommend",
    "info_source_type": "personal_activity_log",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 1,
    "requires_generation": 1,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 1,
    "stakes_level": 2,
    "modality_primary": "structured_log",
    "domain_category": "health_fitness",
    "query_goal_specificity": 1,
    "expected_answer_length": 2,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 1,
    "recommendation_type": "diet_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "summary_text",
    "urgency_phrase_present": 0,
    "planning_horizon": 2,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "budgeting",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 1,
    "habit_analysis": 1,
    "food_context_type": "health_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": 0.2
  },
  {
    "queries": "내가 평소 듣는 콘텐츠 기준으로 재밌는 팟캐스트 추천해줘",
    "QL_char": 33,
    "QL_tokens": 10,
    "task_family": "recommend",
    "info_source_type": "media_history",
    "requires_personal_history": 1,
    "requires_aggregation": 1,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 0,
    "stakes_level": 0,
    "modality_primary": "structured_log",
    "domain_category": "media_entertainment",
    "query_goal_specificity": 0,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 1,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 1,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "content_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 0,
    "planning_horizon": 2,
    "social_context_strength": 0,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "home_desktop",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 1,
    "habit_analysis": 1,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 0,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 1.0,
    "embedding_axis_urgency": -0.1
  },
  {
    "queries": "내가 자주 사는 치약을 최저가로 구매할 수 있는 곳 알려줘",
    "QL_char": 32,
    "QL_tokens": 10,
    "task_family": "find_deal",
    "info_source_type": "external_web",
    "requires_personal_history": 1,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 0,
    "temporal_reference_type": "no_time",
    "time_urgency_level": 1,
    "stakes_level": 2,
    "modality_primary": "mixed/unknown",
    "domain_category": "shopping_retail",
    "query_goal_specificity": 2,
    "expected_answer_length": 0,
    "interaction_pattern": "one_shot_lookup",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 0,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "product_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "simple_fact",
    "urgency_phrase_present": 0,
    "planning_horizon": 1,
    "social_context_strength": 0,
    "contains_monetary_amount": 1,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 1,
    "food_context_type": "spend_optimization",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "financial_analytics",
    "embedding_axis_complexity": 0.9,
    "embedding_axis_urgency": 0.4
  },
  {
    "queries": "나연이랑 갈만한 점심 식당 추천해줘",
    "QL_char": 20,
    "QL_tokens": 7,
    "task_family": "recommend",
    "info_source_type": "external_web",
    "requires_personal_history": 0,
    "requires_aggregation": 0,
    "requires_summarization": 0,
    "requires_generation": 1,
    "temporal_reference_type": "present_now",
    "time_urgency_level": 2,
    "stakes_level": 1,
    "modality_primary": "mixed/unknown",
    "domain_category": "travel_leisure",
    "query_goal_specificity": 1,
    "expected_answer_length": 1,
    "interaction_pattern": "creative_recommendation",
    "has_comparative_phrase": 0,
    "has_temporal_diff_phrase": 0,
    "time_window_length": 0,
    "personalization_intensity": 2,
    "needs_location_context": 1,
    "needs_calendar_context": 0,
    "needs_communication_logs": 0,
    "needs_media_history": 0,
    "needs_financial_logs": 0,
    "needs_health_data": 0,
    "recommendation_type": "place_reco",
    "requires_cross_app_integration": 1,
    "requires_similarity_search": 1,
    "requires_visual_understanding": 0,
    "requires_list_dedup_grouping": 0,
    "explicit_output_structure": "ranked_list",
    "urgency_phrase_present": 1,
    "planning_horizon": 1,
    "social_context_strength": 2,
    "contains_monetary_amount": 0,
    "monetary_context_type": "deal_finding",
    "device_context_implied": "mobile_on_the_go",
    "cognitive_load_estimate": 2,
    "novelty_seeking": 0,
    "habit_analysis": 0,
    "food_context_type": "casual_meal",
    "output_requires_multimedia_creation": 0,
    "output_requires_behavior_change_guidance": 1,
    "embedding_cluster_id": "content_recommendation",
    "embedding_axis_complexity": 0.9,
    "embedding_axis_urgency": 0.9
  }
]
```

---

2. **Feature_Value_Reference**

Below is the reference that should govern all future batches. When I assign values that are heuristic (e.g., embedding-related, device context), they must be treated as **consistent patterns**, not literal numeric truths.

### Core quantitative features

#### `QL_char` (numeric)
- Definition: Count of Unicode characters in the query string, including spaces and punctuation.
- Computation: `len(query)` in Python-style semantics.
- Examples:
  - "지난달에 운동 몇 번이나 했지?" → 18
  - "나연이랑 갈만한 점심 식당 추천해줘" → 20
  - "지금 내가 갖고 있는 멤버십 중에 여기 백화점에서 할인되는 멤버십 찾아줘" → 43
- Use exact character counts per batch; do not approximate.

#### `QL_tokens` (numeric)
- Definition: Number of whitespace-separated tokens.
- Computation: `len(query.split())` (split on spaces).
- Examples:
  - "지난달에 운동 몇 번이나 했지?" → "지난달에","운동","몇","번이나","했지?" → 5
  - "블랙핑크 '뛰어' 랑 비슷한 느낌의 노래 추천해줘" → 8 tokens
  - "내가 자주 사는 치약을 최저가로 구매할 수 있는 곳 알려줘" → 10 tokens

---

### Core categorical / binary / ordinal features

#### `task_family` (categorical)
Values and rules:
- `retrieve_item`: Asking to show/find a specific item or small set: “영상 보여줘”, “어디였지?”, “제목이 뭐였지?”, “멤버십 찾아줘” when it’s essentially lookup, not ranking.
- `count_stats`: Asking for counts/durations: “몇 번이나 했지?”, “총 몇 시간 재생했지?”, “몇 번이나 샀지?”.
- `compare_trend`: Explicit comparison of quantities or trends: contains phrases like “비교해줘”, “평소보다”, “얼마나 늘었지?” across periods/entities.
- `rank_order`: Needs ordering by some metric: “순서대로 알려줘”, “가장 많이 ~ 순서대로”.
- `list_history`: Wants a list of past items/places/events: “리스트 만들어줘”, “모두 정리해서 알려줘”, “다 모아서 쇼핑리스트”.
- `create_media`: Create new media from content (e.g., “영상으로 만들어줘”). None in this sample.
- `recommend`: Suggest items/places/content given constraints: “코스 추천해줘”, “맛집 리스트 보여줘”, “식당 추천해줘”, “노래 추천해줘”.
- `find_deal`: Specifically about cheapest/discount: contains “최저가”, “할인가”, “특가”, “저렴한” and is price-optimization oriented.
- `search_external_info`: Explicit web/news/blog search. Not present here but use for “뉴스 검색해줘”, “블로그 찾아줘”.

Examples:
- "지난주에 찍은 골프 스윙 영상 보여줘" → `retrieve_item`
- "지난달에 운동 몇 번이나 했지?" → `count_stats`
- "온라인 쇼핑과 오프라인 쇼핑 결제 금액 비교해줘" → `compare_trend`
- "올해 가장 많이 들었던 음악 순서대로 알려줘" → `rank_order`
- "스페인 여행에서 내가 갔던 곳 모두 정리해서 알려줘" → `list_history`
- "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘" → `recommend`
- "내가 자주 사는 치약을 최저가로..." → `find_deal`

Boundary: If both listing and recommending, prioritize intent:
- Pure “내 기록 리스트” → `list_history`.
- “기록 기반으로 ~ 추천해줘” → `recommend`.

---

#### `info_source_type` (categorical)
Values:
- `personal_activity_log`: Steps, exercise, calories, activity: “운동”, “걸음 수”, “칼로리 소모량”.
- `personal_content`: User photos, screenshots, memos: “사진”, “찍어둔 것” (if referring to images taken by user).
- `location_history`: Places visited: “… 갔던 곳”, “캠핑장”, “골프장 어디였지?”.
- `calendar_events`: Explicit 일정/회의/세미나/기념일. Use when calendar semantics central.
- `communication_logs`: Calls/messages/chats. Not present here.
- `media_history`: Playback logs: “들었던 음악”, “유튜브 재생”.
- `commerce_history`: Purchases/payments: “구매한”, “결제 금액”, “쇼핑”.
- `external_web`: Generic external info (maps, POI, online shops, music DB) when not clearly from logs.
- `context_now`: Purely “현재 위치”, “지금 여기” without requiring logs. In practice we combine with external_web or logs; in this sample I leave `context_now` unused to avoid fragmentation.

Rules:
- If the query is about past user actions, choose a personal_* or history type.
- If about outside world recommendations (맛집, 코스, 콘텐츠) with no explicit “내 기록”, choose `external_web`.
- Overlaps: pick the **dominant** one. E.g., “현재 위치에서 맛집” → `external_web` (POI search) plus `needs_location_context`=1.

---

#### `requires_personal_history` (binary)
- 1 if the answer clearly depends on *past personal data* (photos taken by user, exercise logs, purchase history, media playback, “내가 자주 ~”, “자주 들었던”, “갔던 곳”).
- 0 otherwise (generic web/POI/content).

Examples:
- "지난달에 운동 몇 번이나 했지?" → 1
- "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘" → 0
- "내가 평소 듣는 콘텐츠 기준으로 재밌는 팟캐스트 추천해줘" → 1

---

#### `requires_aggregation` (binary)
- 1 if needs combining multiple records: counting, summing, listing all, ranking, “모두 모아서”, “순서대로”, “리스트”.
- 0 if single item retrieval or simple recommendation not based on multi-record stats.

Examples:
- "지난달에 운동 몇 번이나 했지?" → 1 (count)
- "올해 가장 많이 들었던 음악 순서대로 알려줘" → 1 (rank by counts)
- "사려고 찍어둔 와인 사진 다 보여줘" → 1 (collect multiple photos)
- "작년 겨울에 구매한 패딩을 어느 사이트에서 샀었지?" → 0 (single purchase)

---

#### `requires_summarization` (binary)
- 1 if explicitly asks to “정리해서”, “요약해서”, “분석해서 ~ 정리해줘”, or conceptually compresses many items into a summary narrative.
- 0 otherwise.

Examples:
- "스페인 여행에서 내가 갔던 곳 모두 정리해서 알려줘" → 1
- "내가 자주 먹는 음식을 분석해서 늦은 밤에 먹기 좋은 건강식 추천해줘" → 1
- Simple lists (“리스트 만들어줘”) without narrative summary → 0.

---

#### `requires_generation` (binary)
- 1 if system must **compose new structured content**: lists, reports, recommendations, shopping lists, etc.
- Typical triggers: “추천해줘”, “리스트 만들어줘”, “쇼핑리스트 만들어줘”, “순서대로 알려줘” (non-trivial list), “정리해서 알려줘”.
- 0 for simple fact retrieval (what/where/which) and raw stat counts.

Examples:
- "이번주 마트에서 ... 쇼핑리스트 만들어줘" → 1
- "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘" → 1
- "지난달에 운동 몇 번이나 했지?" → 0

---

#### `temporal_reference_type` (categorical)
Values:
- `past_specific`: Specific event/day: “어제”, “지난번 여행에서”, “작년 결혼기념일 때”.
- `past_range`: Periods: “지난주”, “지난달”, “올해 상반기”, “작년 겨울”, “작년 이맘때쯤”.
- `present_now`: “오늘”, “이번 주”, “이번주”, “지금”, “현재 위치에서”.
- `future`: “내일”, “다음 주”, etc.
- `no_time`: No explicit time.

Examples:
- "지난주에 찍은..." → `past_range`
- "오늘 목표 걸음수를..." → `present_now`
- "내가 자주 사는 치약을 최저가로..." → `no_time`

---

#### `time_urgency_level` (ordinal: 0,1,2)
Heuristic mapping:
- 2 (high): Contains immediate phrases related to now/very soon decisions: “오늘”, “지금”, “현재 위치”, “점심 식당” (current meal), “이번 주” when planning near-term actions (shopping this week, current week’s calories).
- 1 (medium): Broader but somewhat near-term: “올해”, “상반기”, “만보 코스” (health goal), non-immediate but action-oriented like budgeting, diet planning.
- 0 (low): Pure retrospective stats, old events, reflective habit analysis not tied to imminent decision.

Examples:
- "현재 위치에서 걸어서 갈 수 있는 맛집..." → 2
- "올해 상반기에 옷을 몇 번이나 샀지?" → 1
- "작년 이맘때쯤 자주 들었던 음악 리스트..." → 0

---

#### `stakes_level` (ordinal: 0,1,2)
- 2 (medium-high): Finance and health: money, discounts, budgeting, “결제 금액”, “최저가”, “멤버십 할인”, “건강식”, “운동량”, “칼로리”.
- 1 (medium): Travel/leisure planning, social outings, gifts: “여행”, “캠핑장”, “맛집”, “점심 식당”, “기념일 레스토랑”.
- 0 (low): Entertainment/media, casual history of media/places, curiosity stats.

Examples:
- "온라인 쇼핑과 오프라인 쇼핑 결제 금액 비교해줘" → 2
- "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘" → 2 (health)
- "출퇴근 시간에 볼 만한 영상 추천해줘" → 0

---

#### `modality_primary` (categorical)
Values:
- `photo_image`: Mentions “사진”, “찍어둔 것” clearly as images.
- `video`: “영상” (as media to play or clip).
- `text`: Notes, memos, transcripts. Not present explicitly here.
- `structured_log`: Numeric logs (steps, calories, purchases, playback stats).
- `media_history`: When referring to songs/music as media items (borderline; in this sample I used `structured_log` for logs and reserve `media_history` only where it’s clearly about titles).
- `mixed/unknown`: When both or unclear (e.g., POI search with map, product search).

Examples:
- "사려고 찍어둔 와인 사진 다 보여줘" → `photo_image`
- "지난달에 유튜브를 총 몇 시간 재생했지?" → `structured_log`
- "블랙핑크 '뛰어' 랑 비슷한 느낌의 노래 추천해줘" → `media_history`

---

#### `domain_category` (categorical)
Values:
- `finance`: 결제, 카드, 멤버십 할인, 최저가, 쇼핑 금액.
- `health_fitness`: 운동, 걸음 수, 칼로리, 건강식.
- `media_entertainment`: 음악, 유튜브, 영상, 팟캐스트.
- `social_relationships`: Named people or group context around social events, gifts, anniversaries.
- `travel_leisure`: 여행, 캠핑, 산책 코스, 맛집.
- `shopping_retail`: 마트, 쇼핑, 구매, 물건 리스트.
- `productivity_work`: 회의, 업무, 학습. (Not present here.)
- `fashion_style`: 옷, 신발, 패딩, 패션.
- `other`: None of the above.

Rules:
- Choose the **most salient** domain; e.g., 치약 + 최저가 → `shopping_retail` + finance aspects handled by other features.

---

#### `query_goal_specificity` (ordinal: 0,1,2)
- 2 (highly specific): Clear unique target: specific date/event/person/item: “어디였지?”, “제목이 뭐였지?”, “자주 사는 치약 최저가”, named restaurant, last year’s purchase of a specific item.
- 1 (moderately specific): Has constraints but multiple valid answers: “한강 산책 코스”, “맛집 리스트”, “캠핑장과 비슷한 곳”.
- 0 (very broad): Generic exploration: “재밌는 팟캐스트 추천해줘”, “볼 만한 영상 추천해줘” without strong filters.

Examples:
- "봄에 갔던 골프장 어디였지?" → 2
- "나연이랑 갈만한 점심 식당 추천해줘" → 1
- "내가 평소 듣는 콘텐츠 기준으로 재밌는 팟캐스트 추천해줘" → 0

---

#### `expected_answer_length` (ordinal: 0,1,2)
- 0 (short fact): Single item or scalar: one place, one amount, one yes/no, one count.
- 1 (medium list): Short-to-moderate list of candidates: recommendations, “리스트”, “순서대로” when not huge.
- 2 (long/complex): Extended report, combined explanation, shopping list from many photos, diet plan with explanation.

Examples:
- "작년 결혼기념일 때 갔던 레스토랑 어디였지?" → 0
- "현재 위치에서 걸어서 갈 수 있는 맛집 리스트 보여줘" → 1
- "내가 자주 먹는 음식을 분석해서 ... 건강식 추천해줘" → 2

---

#### `interaction_pattern` (categorical)
Values:
- `one_shot_lookup`: Simple question expecting one fact/number.
- `analytic_report`: Stats, trends, comparisons, structured summaries.
- `creative_recommendation`: Personalized or contextual recommendations (content, places, diet).
- `batch_media_ops`: Operations over many images/media (collecting, filtering) as an operation.

Examples:
- "지난달에 운동 몇 번이나 했지?" → `analytic_report`
- "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘" → `creative_recommendation`
- "사려고 찍어둔 와인 사진 다 보여줘" → `batch_media_ops`

---

### Optional features

#### `has_comparative_phrase` (binary)
- 1 if includes comparison markers: “비교해줘”, “평소보다”, “가장 많이”, “~보다”.
- 0 otherwise.

Examples:
- "온라인 쇼핑과 오프라인 쇼핑 결제 금액 비교해줘" → 1
- "여행 갔을 때 평소보다 평균 걸음 수가 얼마나 늘었지?" → 1
- "지난달에 운동 몇 번이나 했지?" → 0

---

#### `has_temporal_diff_phrase` (binary)
- 1 if explicit temporal comparison: “지난달과 비교했을 때”, “평소보다”, “이번 달과 지난달”, “작년과 올해”.
- 0 otherwise.

Examples:
- "지난달과 비교했을 때 이번 달 운동량이 얼마나 늘었지?" → 1
- "여행 갔을 때 평소보다 평균 걸음 수가 얼마나 늘었지?" → 1

---

#### `time_window_length` (ordinal: 0,1,2)
- 0 (very short): “어제”, “오늘”, “내일”, “이번 주”, “이번주”, “여행 갔을 때” (short trip context).
- 1 (medium): “지난주”, “지난달”, “올해 상반기”, “올해”, “이번 달”.
- 2 (long): “작년”, “지난해”, “작년 이맘때쯤”, “올해 전체” when scanning long periods.

Examples:
- "이번 주 요일 별 칼로리..." → 0
- "지난달에 유튜브를 총 몇 시간..." → 1
- "작년 이맘때쯤 자주 들었던 음악..." → 2

---

#### `personalization_intensity` (ordinal: 0,1,2)
- 0 (generic): No explicit “나/내가/내” patterns; generic constraints only.
- 1 (weak): Light personalization via location/time context: “현재 위치에서”, “집 근처”, “출퇴근 시간에”.
- 2 (strong): Explicit self-referential patterns: “내가 자주 ~”, “내가 평소 듣는”, “내가 갔던 곳”, “내가 갖고 있는 멤버십”, or clearly using personal logs.

Examples:
- "블랙핑크 ... 노래 추천해줘" → 0
- "현재 위치에서 걸어서 갈 수 있는 맛집..." → 1
- "내가 자주 먹는 음식을 분석해서..." → 2

---

#### `needs_location_context` (binary)
- 1 if query references:
  - Explicit location phrases: “현재 위치”, “여기 백화점”, “집 근처”, “한강 산책 코스”, “캠핑장”, “여행 갔을 때”.
- 0 otherwise.

Examples:
- "현재 위치에서 걸어서 갈 수 있는 맛집..." → 1
- "스페인 여행에서 내가 갔던 곳..." → 1
- "지난달에 운동 몇 번이나 했지?" → 0

---

#### `needs_calendar_context` (binary)
- 1 if refers to 일정/캘린더/약속/회의/세미나/기념일.
- 0 otherwise.

Examples:
- "작년 결혼기념일 때 갔던 레스토랑..." → 1
- All others here → 0.

---

#### `needs_communication_logs` (binary)
- 1 if mentions 통화, 메시지, 대화, 채팅, etc.
- 0 otherwise (none in this sample).

---

#### `needs_media_history` (binary)
- 1 if uses playback history: “들었던 음악”, “유튜브를 총 몇 시간 재생했지?”, “가장 많이 들었던”.
- 0 otherwise.

Examples:
- "어제 반복해서 들은 노래..." → 1
- "작년 이맘때쯤 자주 들었던 음악..." → 1

---

#### `needs_financial_logs` (binary)
- 1 if about 결제, 구매한, 쇼핑 금액, 멤버십 할인, 최저가.
- 0 otherwise.

Examples:
- "온라인 쇼핑과 오프라인 쇼핑 결제 금액 비교해줘" → 1
- "내가 자주 사는 치약을 최저가로..." → 0 for logs, but finance is captured via other features (here I set 0, but if explicitly “결제 내역” then 1).

In this sample:
- I set 1 where queries reference commerce history or membership/discounts requiring financial records.

---

#### `needs_health_data` (binary)
- 1 if about 건강, 운동량, 걸음 수, 칼로리, 건강식.
- 0 otherwise.

Examples:
- "이번 주 요일 별 칼로리 소모량..." → 1
- "오늘 목표 걸음수를..." → 1
- "내가 자주 먹는 음식을 분석해서 ... 건강식 추천해줘" → 1

---

#### `recommendation_type` (categorical)
Applies when `task_family` is `recommend` or `find_deal`:
- `content_reco`: Music, videos, podcasts, “영상”, “팟캐스트”.
- `place_reco`: 식당, 카페, 맛집, 여행지, 캠핑장, 산책 코스.
- `product_reco`: Physical products, subscriptions, 치약, 패딩, etc.
- `diet_reco`: 식단, 음식 추천 explicitly for health.
- `gift_reco`: 선물.
- `card_coupon_reco`: 카드, 멤버십, 쿠폰.
- `other`: Recommendation not fitting above.

Examples:
- "출퇴근 시간에 볼 만한 영상 추천해줘" → `content_reco`
- "나연이랑 갈만한 점심 식당 추천해줘" → `place_reco`
- "내가 자주 먹는 음식을 분석해서 ... 건강식 추천해줘" → `diet_reco`
- "지금 내가 갖고 있는 멤버십 중에 ... 할인되는 멤버십 찾아줘" → `card_coupon_reco`

---

#### `requires_cross_app_integration` (binary)
- 1 if fulfilling query clearly needs **multiple** sources:
  - e.g., health logs + food DB, membership list + store DB, personal history + external recommendations.
- 0 if single-log or single external source suffices.

Examples:
- "지금 내가 갖고 있는 멤버십 중에 여기 백화점에서 할인되는 멤버십 찾아줘" → 1 (membership app + store list)
- "내가 자주 먹는 음식을 분석해서 ... 건강식 추천해줘" → 1 (food logs + nutrition DB)
- "지난달에 운동 몇 번이나 했지?" → 0

---

#### `requires_similarity_search` (binary)
- 1 if asks for “비슷한 느낌”, “비슷한 디자인”, “비슷한 곳”, or recommendations explicitly “like X”.
- 0 otherwise.

Examples:
- "블랙핑크 '뛰어' 랑 비슷한 느낌의 노래 추천해줘" → 1
- "이 사진에 있는 신발이랑 비슷한 디자인 찾아줘" → 1
- "지난달에 갔던 캠핑장과 비슷한 곳 찾아줘" → 1

---

#### `requires_visual_understanding` (binary)
- 1 if needs image content understanding: “이 사진에 있는 ~”, “찍어둔 ~ 사진”, “사진만 모아서”.
- 0 otherwise.

Examples:
- "사려고 찍어둔 와인 사진 다 보여줘" → 1
- "지난번 여행에서 ... 찍은 사진 보여줘" → 1
- "이 사진에 있는 신발이랑 비슷한 디자인 찾아줘" → 1

---

#### `requires_list_dedup_grouping` (binary)
- 1 if needs grouping, dedup, ranking over many items:
  - “가장 많이 ~ 순서대로”, “리스트 만들어줘”, “다 모아서 쇼핑리스트”.
- 0 otherwise.

Examples:
- "올해 가장 많이 들었던 음악 순서대로 알려줘" → 1
- "작년 이맘때쯤 자주 들었던 음악 리스트 만들어줘" → 1
- "이번주 마트에서 ... 다 모아서 쇼핑리스트 만들어줘" → 1

---

#### `explicit_output_structure` (categorical)
Values:
- `summary_text`: Narrative/summary: “정리해서 알려줘”, “분석해서 ~ 알려줘”.
- `tabular_report`: Explicit table-like: “가계부로 작성”, “내역 정리”.
- `media_output`: New media creation: “영상으로 만들어줘”.
- `ranked_list`: Ordered or unordered lists: “리스트”, “순서대로 알려줘”.
- `simple_fact`: Single fact/number/location.

Examples:
- "스페인 여행에서 내가 갔던 곳 모두 정리해서 알려줘" → `summary_text`
- "작년 이맘때쯤 자주 들었던 음악 리스트 만들어줘" → `ranked_list`
- "온라인 쇼핑과 오프라인 쇼핑 결제 금액 비교해줘" → `tabular_report`
- "봄에 갔던 골프장 어디였지?" → `simple_fact`

---

#### `urgency_phrase_present` (binary)
- 1 if includes explicit urgency markers: “오늘”, “지금”, “현재 위치에서”, “이번 주” when near-term decision, “점심 식당”.
- 0 otherwise.

Examples:
- "오늘 목표 걸음수를 채울 수 있는 코스..." → 1
- "지금 내가 갖고 있는 멤버십 중에..." → 1
- "현재 위치에서 걸어서 갈 수 있는 맛집..." → 1

---

#### `planning_horizon` (ordinal: 0,1,2)
- 0 (retrospective/analytic): Purely about past (지난달, 작년, 이맘때쯤) with no forward decision.
- 1 (near-future): Immediate/short-term actions: today’s walk, this week’s shopping, lunch, near-term travel.
- 2 (medium-term): Habit change, subscriptions, diet planning, long-term health/finance optimization.

Examples:
- "지난달에 운동 몇 번이나 했지?" → 0
- "이번주 마트에서 사야하는 물건..." → 1
- "내가 자주 먹는 음식을 분석해서 ... 건강식 추천해줘" → 2

---

#### `social_context_strength` (ordinal: 0,1,2)
- 0: No explicit social tie.
- 1: Generic group: 친구들, 가족, 팀원, “결혼기념일” (implied partner but not named).
- 2: Named individual: 나연, 소연, 성진, etc.

Examples:
- "작년 결혼기념일 때 갔던 레스토랑..." → 1
- "나연이랑 갈만한 점심 식당 추천해줘" → 2
- Others without names/groups → 0

---

#### `contains_monetary_amount` (binary)
- 1 if explicit amounts like “5만원”, “10만원 정도”.
- 0 otherwise. (None in this sample except conceptual money; still 0.)

---

#### `monetary_context_type` (categorical)
For money-related queries:
- `spend_tracking`: “어디서 썼지?”, “결제 내역”, “가계부”.
- `budgeting`: “관리비”, “저축”, “용돈”, “생활비”.
- `deal_finding`: “최저가”, “할인가”, “특가”, “할인되는 멤버십”.
- `comparison`: Comparing spend: “결제 금액 비교”, “평소보다 많이 썼나”.

In this sample, I sometimes left defaults even when not strictly monetary; for future batches:
- Only set non-default when money is central.

---

#### `device_context_implied` (categorical)
Heuristic:
- `mobile_on_the_go`:
  - Location phrases: “현재 위치”, “여기 백화점”, “한강 산책 코스”, “집 근처”.
  - Commuting: “출퇴근 시간에”.
  - Immediate in-store or walking context.
- `home_desktop`:
  - Long retrospective analytics: “지난달 카드 결제 내역 정리”, “작년 이맘때쯤 리스트”, “올해 상반기 옷을 몇 번이나 샀지?”.
- `neutral`: Others.

Examples:
- "현재 위치에서 걸어서 갈 수 있는 맛집..." → `mobile_on_the_go`
- "올해 상반기에 옷을 몇 번이나 샀지?" → `home_desktop`
- "블랙핑크 ... 노래 추천해줘" → `neutral`

---

#### `cognitive_load_estimate` (ordinal: 0,1,2)
Based on number of “heavy” operations:
- Consider these as heavy: `requires_aggregation`, `requires_summarization`, `requires_generation`, `requires_cross_app_integration`, `requires_visual_understanding`, `requires_list_dedup_grouping`, `requires_similarity_search`.
- 2 if **≥2** heavy operations are 1.
- 1 if **exactly 1** heavy operation is 1.
- 0 if none are 1.

Examples:
- "지난달에 운동 몇 번이나 했지?" → `requires_aggregation`=1 only → 1
- "이번주 마트에서 ... 쇼핑리스트 만들어줘" → aggregation + generation + visual_understanding + list_dedup + cross_app → 2
- "봄에 갔던 골프장 어디였지?" → none → 0

---

#### `novelty_seeking` (binary)
- 1 if explicitly seeks “새로운”, “최신”, “재밌는 ~ 추천해줘” without strong specificity, or “볼 만한” in a discovery sense.
- 0 otherwise.

Examples:
- "블랙핑크 ... 비슷한 느낌의 노래 추천해줘" → 1
- "출퇴근 시간에 볼 만한 영상 추천해줘" → 1
- "내가 평소 듣는 콘텐츠 기준으로 재밌는 팟캐스트 추천해줘" → 1

---

#### `habit_analysis` (binary)
- 1 if about habits/patterns: “자주 ~”, “가장 많이 ~ 했던”, “평소보다”, “분석해서”.
- 0 otherwise.

Examples:
- "작년 이맘때쯤 자주 들었던 음악 리스트..." → 1
- "올해 가장 많이 들었던 음악..." → 1
- "내가 자주 먹는 음식을 분석해서..." → 1

---

#### `food_context_type` (categorical)
For food-related queries:
- `health_meal`: Health/late-night/diet context: “건강식”, “늦은 밤에 먹기 좋은”.
- `casual_meal`: “점심 메뉴”, “맛집”, generic dining.
- `spend_optimization`: Coffee beans, subscriptions, cost-focused.

In this sample:
- "내가 자주 먹는 음식을 분석해서 늦은 밤에 먹기 좋은 건강식..." → `health_meal`.
- "나연이랑 갈만한 점심 식당 추천해줘" → `casual_meal`.
- 치약 최저가 → `spend_optimization` (spending context, though not food in a narrow sense).

---

#### `output_requires_multimedia_creation` (binary)
- 1 if explicitly requests new media output: “영상으로 만들어줘”, “슬라이드쇼로 만들어줘”.
- 0 otherwise (all rows here).

---

#### `output_requires_behavior_change_guidance` (binary)
- 1 if output is guidance for behavior change: exercise routines, diet, subscription choices, shopping optimization, financial/health behavior.
- 0 otherwise.

Examples:
- "만보 정도 걸을 수 있는 한강 산책 코스 추천해줘" → 1
- "오늘 목표 걸음수를 채울 수 있는 코스 추천해줘" → 1
- "내가 자주 먹는 음식을 분석해서 ... 건강식 추천해줘" → 1
- "내가 자주 사는 치약을 최저가로..." → 1
- Simple retrospective stats (“지난달에 유튜브 총 몇 시간...”) → 0.

---

#### `embedding_cluster_id` (categorical, proxy here)
Since we don’t actually have embeddings, I used **semantic cluster labels** as if derived from embeddings. For consistency across batches, use these labels:

- `"personal_media_retrieval"`: Queries retrieving personal photos/videos or specific past places.
- `"activity_stats"`: Health/fitness/media usage statistics and history lists.
- `"financial_analytics"`: Commerce/finance-related queries (purchases, deals, memberships).
- `"content_recommendation"`: Content/POI/product recommendations (music, videos, places, diet).
  
For future rows:
- Assign the cluster most semantically similar to these patterns.

---

#### `embedding_axis_complexity` (numeric, 0–1 scale)
Conceptual continuous proxy:
- ~0.0–0.3: Simple one-shot lookups.
- ~0.4–0.6: Single-operation analytics (counts, simple retrieval with logs).
- ~0.7–1.0: Multi-operation analytics, recommendations with personalization, cross-app integration, or visual understanding.

Rules:
- Start from `cognitive_load_estimate`:
  - 0 → ~0.2–0.3
  - 1 → ~0.4–0.7
  - 2 → ~0.8–1.0
- Slightly adjust within band for length and personalization.

Consistency: For future batches, follow this mapping pattern; exact numeric is less important than relative ordering.

---

#### `embedding_axis_urgency` (numeric, approx -0.3 to 1.0)
Conceptual urgency proxy:
- Negative (~-0.3 to 0): Purely retrospective, low urgency.
- Around 0: Neutral.
- 0.3–0.6: Some urgency (near-term but not “now”).
- 0.7–1.0: Explicit “오늘/지금/현재 위치” immediate decisions.

Rules:
- Base on `time_urgency_level` and `urgency_phrase_present`:
  - time_urgency_level 0 & no urgency phrase → -0.3 to 0.
  - level 1 → ~0.2–0.4.
  - level 2 & urgency_phrase_present=1 → ~0.7–1.0.

---

3. **Modeling_Notes**

- Likely most predictive features (for MAL):
  - **Urgency-related**: `time_urgency_level`, `temporal_reference_type`, `urgency_phrase_present`, `planning_horizon`, `embedding_axis_urgency`. High urgency should correspond to **lower MAL**.
  - **Complexity-related**: `task_family`, `requires_aggregation`, `requires_summarization`, `requires_generation`, `requires_cross_app_integration`, `requires_visual_understanding`, `requires_list_dedup_grouping`, `cognitive_load_estimate`, `embedding_axis_complexity`. Higher complexity tends to increase **acceptable** latency.
  - **Stakes-related**: `stakes_level`, `domain_category`, `needs_financial_logs`, `needs_health_data`, `contains_monetary_amount`. Higher stakes usually reduce MAL (users want fast/accurate answers).
  - **Personalization & history**: `requires_personal_history`, `personalization_intensity`, `info_source_type`. Deep personalized analytics may let users tolerate slightly more latency.

- Recommended model families:
  - Start with **regularized linear models** (Lasso/Elastic Net) on the core set plus a few optional features (`cognitive_load_estimate`, `planning_horizon`, `personalization_intensity`) for interpretability.
  - Add a **shallow tree or small Random Forest** to capture non-linearities (e.g., high urgency × high stakes interactions).
  - If you want strict interpretability, a **GLM with interactions** like `stakes_level × time_urgency_level` and `cognitive_load_estimate × requires_personal_history` is reasonable.

- Evaluation plan:
  - Use **5-fold cross-validation** on the 256 rows to avoid overfitting.
  - Metrics: **MAE** (primary, intuitive in seconds), **RMSE** (penalize large errors), and **R²**.
  - Optionally bucket MAL into low/medium/high and track **bucket accuracy** or macro-F1 to see if the model gets the “class” of latency right.

---

4. **Interpretation_Summary**

- The features are designed to reflect three main human-factors drivers of acceptable latency:
  - **Urgency & stakes** (time-critical decisions and money/health consequences) push MAL downward: users want quick results when deciding lunch now, picking a discount at the store, or adjusting today’s steps.
  - **Cognitive/technical complexity** (aggregation, summarization, cross-app integration, visual understanding) pushes MAL upward: users expect that mining logs, analyzing habits, or building ranked lists “takes time” and are more tolerant of delay.
  - **Personalization depth & engagement** (personal history, named social ties, habit analysis) often increases users’ willingness to wait slightly longer for high-quality, meaningful answers.

- Together, these features let a model distinguish, for example, a quick “어디였지?” lookup (low complexity, moderate stakes, low MAL) from a cross-app, visual, habit-analysis recommendation (high complexity, reflective, higher MAL), even if both queries are similar in length.