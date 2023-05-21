select
"marketing_source",
"create_date",
"known_city",
"message_length",
"test_flag",
"customer_date",
"contract_length",
"trial_booked",
"trial_date",
"call_attempts",
"total_call_duration",
"calls_30",
CASE WHEN contract_length is not null then 1 else 0 end as "target_column",
CASE WHEN create_date::DATE <= current_date - 90 then 'train' else 'test' end as "is_train"
from
    public.lead_dataset
left join public.customer_dataset on lead_dataset.contact_id = customer_dataset.contact_id
left join call_dataset on call_dataset.contact_id = lead_dataset.contact_id