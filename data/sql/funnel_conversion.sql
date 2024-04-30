select
date_trunc('week', create_date::DATE)::DATE as week_creation,
count(distinct lead_dataset.contact_id) as leads,

count(distinct CASE WHEN
            NULLIF(call_dataset.call_attempts, '')::FLOAT > 0
    then
    call_dataset.contact_id
    end) as contact_attempted,
count(distinct CASE WHEN
            NULLIF(call_dataset.calls_30, '')::FLOAT > 0
    then
    call_dataset.contact_id
    end) as contacted,
count(distinct customer_dataset.contact_id) as customers,
avg(clv_dataset.avg_clv::float) as avg_customer_value,
avg(CASE WHEN
            NULLIF(call_dataset.call_attempts, '')::FLOAT > 0
    then
    NULLIF(call_dataset.call_attempts, '')::FLOAT
    end) as avg_contact_attempted

from
    public.lead_dataset
left join public.customer_dataset on lead_dataset.contact_id = customer_dataset.contact_id
left join call_dataset on call_dataset.contact_id = lead_dataset.contact_id
left join clv_dataset on customer_dataset.contract_length = clv_dataset.contract_length
group by 1
order by 1