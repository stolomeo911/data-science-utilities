
with leads as (select
date_trunc('month', customer_date::DATE)::DATE as month_creation,
sum(clv_dataset.avg_clv::float) as sum_customer_value

from
    public.lead_dataset
left join public.customer_dataset on lead_dataset.contact_id = customer_dataset.contact_id
left join call_dataset on call_dataset.contact_id = lead_dataset.contact_id
left join clv_dataset on customer_dataset.contract_length = clv_dataset.contract_length
group by 1
order by 1,2)

SELECT
sales_cost_dataset."month",
sales_cost_dataset."total_sales_costs",
sum_customer_value::float / sales_cost_dataset."total_sales_costs"::float as "roi_sales_costs"
FROM
public.sales_cost_dataset
LEFT JOIN leads on to_date(sales_cost_dataset.month, 'YYYY-MM') = leads.month_creation::date
order by 1, 2