select
lead_dataset.id,
lead_dataset.marketing_source,
create_date,
known_city,
message_length,
test_flag,
case when customer_dataset.id is not null then 1 else 0 end as "has_converted",
clv_dataset.contract_length,
clv_dataset.avg_clv,
trial_booked,
call_attempts,
calls_30

from lead_dataset
         left join public.customer_dataset on lead_dataset.contact_id = customer_dataset.contact_id
left join call_dataset on call_dataset.contact_id = lead_dataset.contact_id
left join clv_dataset on customer_dataset.contract_length = clv_dataset.contract_length
