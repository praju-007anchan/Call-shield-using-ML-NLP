import { Activity, BarChart3, ChevronRight, CircleHelp, ShieldCheck, Wifi } from 'lucide-react';
import { Link, useLocation } from 'wouter';
import { useGetDashboard, useHealthCheck } from '@workspace/api-client-react';

export function CallShell({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();
  const { data: dashboard, isLoading: dashboardLoading } = useGetDashboard();
  const { data: health } = useHealthCheck();
  const isAnalyze = location === '/';

  return (
    <div className="min-h-[100dvh] bg-background md:flex">
      <aside className="flex w-full shrink-0 flex-col bg-sidebar text-sidebar-foreground md:sticky md:top-0 md:h-[100dvh] md:w-[236px]">
        <div className="flex items-center justify-between px-5 py-5 md:block md:px-6 md:pt-7">
          <Link href="/" data-testid="link-brand" className="flex items-center gap-3 text-sidebar-foreground">
            <span className="grid h-8 w-8 place-items-center rounded-[10px] bg-primary text-primary-foreground">
              <ShieldCheck size={19} strokeWidth={2.5} />
            </span>
            <span className="text-[17px] font-semibold tracking-[-0.04em]">CallShield</span>
          </Link>
          <span className="hidden font-mono text-[9px] uppercase tracking-[0.2em] text-sidebar-foreground/45 md:block md:pl-11">review workspace</span>
        </div>

        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:mt-12 md:block md:space-y-1 md:px-3" aria-label="Main navigation">
          <NavItem href="/" active={isAnalyze} icon={<Activity size={16} />} label="Analyze call" testId="link-analyze" />
          <NavItem href="/models" active={location === '/models'} icon={<BarChart3 size={16} />} label="Model bench" testId="link-models" />
        </nav>

        <div className="hidden md:block md:flex-1" />
        <div className="hidden border-t border-sidebar-border px-5 py-5 md:block">
          <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.13em] text-sidebar-foreground/50">
            <Wifi size={12} className={health?.status === 'ok' ? 'text-primary' : 'text-sidebar-foreground/50'} />
            System status
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-sidebar-foreground/75">{health?.status === 'ok' ? 'API connected' : 'Checking API'}</span>
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
          </div>
        </div>
        <div className="hidden items-center gap-2 border-t border-sidebar-border px-5 py-4 text-sidebar-foreground/45 md:flex">
          <CircleHelp size={15} />
          <span className="text-xs">Low-confidence by design</span>
        </div>
      </aside>

      <main className="min-w-0 flex-1">
        <header className="flex h-[68px] items-center justify-between border-b border-border bg-background/90 px-5 backdrop-blur md:px-9">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Call intelligence / <span className="text-foreground">{isAnalyze ? 'live review' : 'model bench'}</span></p>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden text-right sm:block">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">calls reviewed</p>
              <p className="text-sm font-semibold">{dashboardLoading ? '—' : (dashboard?.callsAnalyzed ?? 0).toLocaleString()}</p>
            </div>
            <div className="grid h-8 w-8 place-items-center rounded-full bg-secondary text-xs font-semibold text-secondary-foreground" data-testid="avatar-reviewer">AR</div>
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}

function NavItem({ href, active, icon, label, testId }: { href: string; active: boolean; icon: React.ReactNode; label: string; testId: string }) {
  return (
    <Link href={href} data-testid={testId} className={`group flex min-w-max items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors md:w-full ${active ? 'bg-sidebar-accent text-sidebar-accent-foreground' : 'text-sidebar-foreground/60 hover:bg-sidebar-accent/70 hover:text-sidebar-foreground'}`}>
      {icon}
      <span>{label}</span>
      <ChevronRight size={14} className={`ml-auto transition-opacity ${active ? 'opacity-60' : 'opacity-0 group-hover:opacity-40'}`} />
    </Link>
  );
}