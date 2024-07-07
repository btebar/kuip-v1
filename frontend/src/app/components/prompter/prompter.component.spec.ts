import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PrompterComponent } from './prompter.component';

describe('PrompterComponent', () => {
  let component: PrompterComponent;
  let fixture: ComponentFixture<PrompterComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PrompterComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PrompterComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
