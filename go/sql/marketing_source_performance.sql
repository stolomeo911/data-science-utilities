with leads as (select
date_trunc('month', create_date::DATE)::DATE as month_creation,
lead_dataset.marketing_source,
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
sum(clv_dataset.avg_clv::float) as sum_customer_value

from
    public.lead_dataset
left join public.customer_dataset on lead_dataset.contact_id = customer_dataset.contact_id
left join call_dataset on call_dataset.contact_id = lead_dataset.contact_id
left join clv_dataset on customer_dataset.contract_length = clv_dataset.contract_length
group by 1,2
order by 1,2)

SELECT
marketing_costs_dataset."date",
marketing_costs_dataset."marketing_source",
"marketing_costs_dataset"."marketing_costs",
marketing_costs::float / leads::float  as "cost_per_lead",
marketing_costs::float / customers::float as "cost_per_acquisition",
sum_customer_value::float / marketing_costs::float as "roi_marketing_source"
FROM
public.marketing_costs_dataset
LEFT JOIN leads on to_date(marketing_costs_dataset.date, 'YYYY-MM') = leads.month_creation::date
                       AND marketing_costs_dataset.marketing_source = leads.marketing_source
order by 1, 2